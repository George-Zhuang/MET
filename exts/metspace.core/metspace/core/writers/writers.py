__copyright__ = "Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import asyncio
import inspect
import io
import json
import uuid
from abc import ABC, abstractmethod
from builtins import NotImplemented, NotImplementedError
from functools import partial
from typing import Callable, Dict, Iterable, List, Tuple, Union

import carb
import omni.graph.core as og
import omni.kit
import omni.usd
from omni.replicator.core.bindings._omni_replicator_core import Schema_omni_replicator_extinfo_1_0
from omni.syntheticdata import SyntheticData
from pxr import Sdf, Usd

from .annotators import Annotator, AnnotatorRegistry
from .trigger import on_condition
from .utils.utils import (
    ReplicatorItem,
    ReplicatorWrapper,
    auto_connect,
    create_node,
    get_exec_attr,
    get_reduced_ref_time,
    send_og_event,
)
from .utils.viewport_manager import HydraTexture

GRAPH_PATH = "/Render/PostProcess/SDGPipeline"
DEFAULT_WRITERS = ["BasicWriter", "KittiWriter", "FPSWriter"]


def _connect_execs(upstream_node, downstream_node):
    """Connect first upstream exec attribute to first downstream exec attribute"""
    upstream_exec, downstream_exec = None, None
    for attr in upstream_node.get_attributes():
        if attr.get_port_type() != og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT:
            continue
        if attr.get_resolved_type().get_role_name() == "execution":
            upstream_exec = attr
            break
    for attr in downstream_node.get_attributes():
        if attr.get_port_type() != og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT:
            continue
        if attr.get_resolved_type().get_role_name() == "execution":
            downstream_exec = attr
            break
    if upstream_exec and downstream_exec:
        upstream_exec.connect(downstream_exec, True)


def _create_node_attribute(node, attribute, dtype):
    if node.get_attribute_exists(attribute):
        return
    node.create_attribute(attribute, dtype)


def _set_node_attributes(node, attributes: dict):
    for attr, value in attributes.items():
        if not node.get_attribute_exists(attr):
            continue
        og.AttributeValueHelper(node.get_attribute(attr)).set(value, update_usd=True)


def _get_or_create_node(graph, node_type_id, node_name, attributes=None):
    controller = og.Controller()
    graph_path = graph.get_path_to_graph()
    node = og.get_node_by_path(f"{graph_path}/{node_name}")
    if node is None:
        node = controller.create_node((node_name, graph), node_type_id)

    if attributes:
        _set_node_attributes(node, attributes)
    return node


def _connect_attributes(src_node, dst_node, src_attr, dst_attr):
    if len(src_attr) != len(dst_attr):
        raise ValueError

    for src_attr_name, dst_attr_name in zip(src_attr, dst_attr):
        if not src_node.get_attribute_exists(src_attr_name) or not dst_node.get_attribute_exists(dst_attr_name):
            continue
        src_attr = src_node.get_attribute(src_attr_name)
        dst_attr = dst_node.get_attribute(dst_attr_name)
        if not src_attr.is_connected(dst_attr):
            src_attr.connect(dst_attr, True)


def _connect_to_writer(graph, sync_node, writer_node, annotator):
    render_product = [annotator._render_products[rpi] for rpi in annotator._render_product_idxs][
        0
    ]  # annotator can be associated with only one render product
    annotator_node = annotator.get_node()
    annotator_name = annotator._name
    with Sdf.ChangeBlock():
        for attr in annotator_node.get_attributes():
            if attr.get_port_type() == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT and "outputs" in attr.get_name():
                # If annotator execution attribute
                if attr.get_resolved_type().get_role_name() == "execution":
                    # link to async gate input execution
                    _connect_attributes(annotator_node, sync_node, [attr.get_name()], ["inputs:execIn"])
                    continue

                attr_name = attr.get_name()[8:]

                writer_attr_name = f"inputs:{render_product.split('/')[-1]}:{annotator_name}:{attr_name}"

                _create_node_attribute(writer_node, writer_attr_name, attr.get_resolved_type())
                _connect_attributes(annotator_node, writer_node, [attr.get_name()], [writer_attr_name])


class WriterRegistryError(Exception):
    """Basic exception for errors raised by the writer registry"""

    def __init__(self, msg=None):
        if msg is None:
            msg = "A WriterRegistry error was encountered."
        super().__init__(msg)


class WriterError(Exception):
    """Basic exception for errors raised by a writer"""

    def __init__(self, msg=None):
        if msg is None:
            msg = "A Writer error was encountered."
        super().__init__(msg)


class InvalidWriterError(WriterRegistryError):
    """Exception for writer registration errors"""

    def __init__(self, writer_name, writer, msg=None):
        if msg is None:
            msg = f"The writer `{writer_name}` is invalid."
        super().__init__(msg)
        self.writer_name = writer_name
        self.writer = writer


class NodeWriter:
    """Node Writer class.

    Node writers are writers implemented as OmniGraph nodes. These depend on annotators like python writers, but
    are implemented as nodes and can be written in C++.

    Args:
        node_type_id: The node's type identifier (eg. `'my.extension.OgnCustomNode'`)
        annotators: List of dependent annotators

    Attributes:
        node_type_id: The node's type identifier (eg. `'my.extension.OgnCustomNode'`)
        annotators: List of dependent annotators
        kwargs: Node Writer input attribute initialization
    """

    def __init__(self, node_type_id: str, annotators: List[Union[str, Annotator]], **kwargs):
        self.node_type_id = node_type_id
        self._annotators = annotators
        self._kwargs = kwargs
        self._node = None

    def initialize(self, **kwargs) -> None:
        self._kwargs = dict(self._kwargs, **kwargs)

    def __call__(self, **kwargs):
        self.initialize(**kwargs)
        return self

    @property
    def annotators(self) -> List[Union[str, Annotator]]:
        return self._annotators

    def get_node(self):
        """Get writer node"""
        if self._node:
            return self._node
        else:
            raise InvalidWriterError(self.node_type_id, self, "Unable to retrieve writer node, writer is not attached.")

    def attach(self, render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]], **kwargs):
        """Attach node writer to render products

        Attaches the node writer to the render products specified and creates and attaches annotator graphs. If
        a single annotator is attached, input connections will match the annotator's output attribute names, in the
        form of `inputs:<attribute_name>`. For cases involving more than one annotators, the node writer input
        attributes will be in the form `inputs:<annotator_name>_<attribute_name>`.

        Args:
            render_products: Render Product HydraTexture object(s) or prim path(s) to which to attach the writer.
            kwargs: Node Writer input attribute initialization
        """
        init_params = dict(self._kwargs, **kwargs)
        WriterRegistry.attach(self, render_products, trigger="omni.replicator.core.OgnOnFrame", **init_params)
        # Send writer attached event
        message_bus = omni.kit.app.get_app().get_message_bus_event_stream()
        message_bus.dispatch(
            carb.events.type_from_string("omni.replicator.core.writers"), payload={"attached": self.node_type_id}
        )

    def on_final_frame(self):
        """Run after final frame is written."""
        pass

    def detach(self):
        """Detach writer"""
        WriterRegistry._detach_by_writer(self)


class Writer(ABC):
    """Base Writer class.

    Writers must specify a list of required annotators which will be called during data collection. Annotator data is
    packaged in a `data` dictionary of the form `<annotator_name>: <annotator_data>` and passed to the writer's `write`
    function.

    An optional `on_final_frame` function can be defined to run once data generation is stopped.
    """

    backend = None
    version = "0.0.0"
    annotators = []
    metadata = None
    num_written = 0
    _is_metadata_written = False
    _is_warning_version_posted = False
    _is_warning_backend_posted = False
    __data = None
    __schedule = None
    __max_schedule_len = 10

    def get_metadata(self):
        """Get writer metadata"""
        if not self._is_warning_version_posted and self.version == "0.0.0":
            carb.log_warn("Writer version not specified in the writer metadata")
            self._is_warning_version_posted = True

        self.metadata = {
            "name": self.__class__.__name__,  # this is the subclass i.e. the writer name
            "version": self.version,
        }

    def write_metadata(self):
        # this saves the metadata that will be used to read the file
        if self.backend is None:
            if not self._is_warning_backend_posted:
                carb.log_warn(
                    "Unable to write metadata, no backend specified. To enable metadata writing, "
                    "either specify a backend or override the `write_metadata` function."
                )
                self._is_warning_backend_posted = True
            return
        buf = io.StringIO()
        buf.write(json.dumps(self.metadata))
        self.backend.write_blob("metadata.txt", bytes(buf.getvalue(), "utf-8"))
        self._is_metadata_written = True

    @abstractmethod
    def write(self, data: dict):
        """Write ground truth."""
        raise NotImplementedError

    def schedule_write(self):
        """Manually schedule a write call to the writer

        Sends a "writerTrigger" event to schedule the writer for the current simulation frame. Used in conjunction with
        writer trigger set to `None` (ie. `writer.attach(<render_product>, trigger=None`).

        NOTE: The writer will not write data until the scheduled frame is rendered through subsequent `update` or
            `step` calls.

        Example:
            >>> import omni.replicator.core as rep
            >>> rp = rep.create.render_product(rep.create.camera(), (512, 512))
            >>> writer = rep.writers.get(
            ...     name="BasicWriter",
            ...     init_params={"output_dir": "_out", "rgb": True},
            ...     render_products=rp,
            ...     trigger=None,
            ... )
            >>> # ... initialize/step orchestrator
            >>> writer.schedule_write()
            >>> # ... step/update to render and write scheduled frame
        """
        send_og_event(f"writerTrigger-{self._writer_id}")

    def _schedule(self, ref_time):
        if self.__schedule is None:
            self.__schedule = []
        self.__schedule.append(ref_time)
        while len(self.__schedule) > self.__max_schedule_len:
            self.__schedule.pop(0)

    def _write(self, data: dict):
        if not self._is_metadata_written:
            self.get_metadata()
            self.write_metadata()

        # Cache data so writer can be manually triggered
        self.__data = data

        # If writer is scheduled, call write
        if self.__data is None or not self.__schedule:
            return
        if get_reduced_ref_time(*data["reference_time"]) in self.__schedule:
            self.write(data)
            self.num_written += 1

    def get_data(self) -> Dict:
        """Get the writer's current data payload.

        Returns the data payload currently stored in the writer. Note that this payload corresponds to the frame at
        `data["reference_time"]` which may be older than the current simulation time. Use `rep.orchestrator.step_async`
        to ensure that the simulation state matches the writer payload data.
        """
        return self.__data

    def on_final_frame(self):
        """Run after final frame is written."""
        if self.__class__.__name__ in DEFAULT_WRITERS:
            WriterRegistry._telemetry.writer_sendEvent(self.__class__.__name__, self.num_written)
        else:
            WriterRegistry._telemetry.writer_sendEvent("Custom Writer", self.num_written)

    def initialize(self, **kwargs):
        """Initialize writer
        If the writer takes initialization arguments, they can be set here.

        Args:
            **kwargs: Writer initialization arguments.
        """
        self.__init__(**kwargs)

    async def __attach_async(self, render_products, trigger: Union[ReplicatorItem, Callable]):
        while any([not rp.done() for rp in render_products if asyncio.isfuture(rp)]):
            await omni.kit.app.get_app().next_update_async()

        render_products_results = [rp.result() if asyncio.isfuture(rp) else rp for rp in render_products]
        render_products = []
        while render_products_results:
            rp_r = render_products_results.pop(0)
            if not isinstance(rp_r, List):
                rp_r = [rp_r]

            for rp in rp_r:
                if isinstance(rp, (str, Sdf.Path)):
                    render_products.append(str(rp))
                elif isinstance(rp_r, (str, HydraTexture)):
                    render_products.append(rp.path)
                else:
                    raise ValueError(f"Received invalid render product of type `{type(rp)}`")

        WriterRegistry.attach(self, render_products, trigger)

    def attach(
        self,
        render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]],
        trigger: Union[ReplicatorItem, Callable] = "omni.replicator.core.OgnOnFrame",
    ):
        """Attach writer to specified render products.

        Args:
            render_products: Render Product prim path(s) to which to attach the writer.
            trigger: Function or replicator trigger that triggers the `write` function of the writer. If a function
                is supplied, it must return a boolean. If set to `None`, the writer is set to a manual mode
                where it can be triggered by calling `writer.schedule_write`.
        """
        # print('DEBUG: Writer.attach()')
        if isinstance(render_products, (str, HydraTexture)) or isinstance(render_products, asyncio.Task):
            render_products = [render_products]

        if any([asyncio.isfuture(rp) for rp in render_products]):
            asyncio.ensure_future(self.__attach_async(render_products))
        else:
            WriterRegistry.attach(self, render_products, trigger=trigger)

        self.num_written = 0

    async def attach_async(
        self,
        render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]],
        trigger: Union[ReplicatorItem, Callable] = "omni.replicator.core.OgnOnFrame",
    ):
        """Attach writer to specified render products and await attachment completed

        Args:
            render_products: Render Product prim path(s) to which to attach the writer.
            trigger: Function or replicator trigger that triggers the `write` function of the writer. If a function
                is supplied, it must return a boolean. If set to `None`, the writer is set to a manual mode
                where it can be triggered by calling `writer.schedule_write`.
        """
        # print('DEBUG: Writer.attach_async()')
        if isinstance(render_products, (str, HydraTexture)) or isinstance(render_products, asyncio.Task):
            render_products = [render_products]

        if any([asyncio.isfuture(rp) for rp in render_products]):
            await self.__attach_async(render_products, trigger=trigger)
        else:
            WriterRegistry.attach(self, render_products, trigger=trigger)

    def detach(self):
        """Detach writer"""
        WriterRegistry._detach_by_writer(self)

    def get_node(self) -> og.Node:
        """Get writer node"""
        if self._node:
            return self._node
        else:
            raise InvalidWriterError(
                self.__class__.__name__, self, "Unable to retrieve writer node, writer is not attached."
            )


class WriterRegistry:
    """
    This class stores a list of available registered writers and which writers are attached to render products.

    One or more writers can be attached simultaneously to the same render product to simultaneously write ground truth
    in multiple formats.

    Register a writer with `WriterRegistry.register(ExampleWriter)`
    Attach a writer to a render_product with `WriterRegistry.attach("Example", "/World/Render Product")`
    Detach a writer with `WriterRegistry.detach("Example")`
    """

    _writers = {}
    _categories = {}
    _active_writers = {}
    _telemetry = Schema_omni_replicator_extinfo_1_0()
    _default_writers = DEFAULT_WRITERS

    @classmethod
    def register(cls, writer: Writer, category: str = None) -> None:
        """Register a writer.

        Registered writers can be retrieved with `WriterRegistry.get(<writer_name>)`

        Args:
            writer: Instance of class `Writer`.
            category: Optionally specify a category of writer to group writers together.
        """

        writer_name = writer.__name__
        if not issubclass(writer, Writer):
            raise InvalidWriterError(writer_name, writer, f"Writer must be of class `Writer`, got {type(writer)}")
        if not isinstance(writer.annotators, Iterable):
            msg = f"Writer annotators must be specified as a list of annotators. Got {type(writer.annotators)}"
            raise InvalidWriterError(writer_name, writer, msg)
        if writer_name in cls._writers:
            carb.log_warn(f"Writer already exists. Overwriting writer {writer_name}.")
        cls._writers[writer_name] = writer

        if category:
            cls._categories.setdefault(category, []).append(writer_name)

    @classmethod
    def register_node_writer(
        cls, name: str, node_type_id: str, annotators: List[Union[str, Annotator]], category: str = None, **kwargs
    ) -> None:
        """Register a Node Writer

        Register a writer implemented as an omnigraph node.

        Args:
            node_type_id: The node's type identifier (eg. `'my.extension.OgnCustomNode'`)
            annotators: List of dependent annotators
            category: Optionally specify a category of writer to group writers together.
            kwargs: Node Writer input attribute initialization
        """
        if not isinstance(name, str):
            raise WriterRegistryError(f"Invalid writer name `{name}` of type `{type(name)}`.")
        if not all([isinstance(a, (str, Annotator, SyntheticData.NodeConnectionTemplate)) for a in annotators]):
            raise WriterRegistryError(f"Got one or more invalid annotators in {annotators}.")
        if name in cls._writers:
            carb.log_warn(f"Writer already exists. Overwriting writer {name}.")
        cls._writers[name] = NodeWriter(node_type_id=node_type_id, annotators=annotators, **kwargs)

        if category:
            cls._categories.setdefault(category, []).append(name)

    @classmethod
    def unregister(cls, writer_name) -> None:
        """
        Unregister a writer with specified name if it exists.

        Args:
            writer_name: Name of registered writer.
        """
        if writer_name not in cls._writers:
            raise WriterRegistryError(f"No writer with name `{writer_name}` was found in registry.")
        else:
            cls.detach(writer_name)
            del cls._writers[writer_name]
            for category, writers_names in cls._categories.items():
                if writer_name in writers_names:
                    cls._categories[category].pop(cls._categories[category].index(writer_name))

    @classmethod
    def attach(
        cls,
        writer: Union[Writer, NodeWriter],
        render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]],
        init_params: Dict = None,
        trigger: Union[ReplicatorItem, Callable] = "omni.replicator.core.OgnOnFrame",
        **kwargs,
    ) -> None:
        """
        Attach writer with specified name to render_products.

        Args:
            writer: Writer instance
            render_products: Render Product prim path(s).
            init_params: (Deprecated) Dictionary of initialization parameters.
            trigger: Function or replicator trigger that triggers the `write` function of the writer. If a function
                is supplied, it must return a boolean. If set to `None`, the writer is set to a manual mode
                where it can be triggered by calling `writer.schedule_write`.
            kwargs: Additional initialization parameters.
        """
        # print(f'DEBUG: WriterRegistry.attach() - 00')
        if isinstance(render_products, (str, HydraTexture)):
            render_products = [render_products]
        elif isinstance(render_products, HydraTexture):
            render_products = [render_products.path]

        # print(f'DEBUG: WriterRegistry.attach() - 01')
        # Get render product paths
        render_products = [rp.path if isinstance(rp, HydraTexture) else rp for rp in render_products]

        if init_params:
            carb.log_warn("`init_params` will be deprecated. Initialization parameters can now be set using kwargs")
        elif init_params is None:
            init_params = {}
        # print(f'DEBUG: WriterRegistry.attach() - 02')

        writer_name = writer.__class__.__name__
        active_writer_id = f"{writer_name}_{str(uuid.uuid1())}"
        combined_kwargs = dict(init_params, **kwargs)
        stage = omni.usd.get_context().get_stage()
        if stage:
            session_layer = stage.GetSessionLayer()
            with Usd.EditContext(stage, session_layer):
                # print(f'DEBUG: WriterRegistry.attach() - 03')
                writer_node = cls._attach(writer, active_writer_id, render_products, **combined_kwargs)
                # print(f'DEBUG: WriterRegistry.attach() - 04')
        else:
            raise WriterRegistryError("Invalid USD stage, unable to attach writer")
        for render_product in render_products:
            # print(f'DEBUG: WriterRegistry.attach() - 05')
            cls._active_writers.setdefault((render_product,), {})[active_writer_id] = writer
            # print(f'DEBUG: WriterRegistry.attach() - 06')

            if writer_name in cls._default_writers:
                cls._telemetry.writer_sendEvent(writer_name, -1)
            else:
                cls._telemetry.writer_sendEvent("Custom writer", -1)
        writer._writer_id = active_writer_id
        writer._node = writer_node
        # print(f'DEBUG: WriterRegistry.attach() - 07')

        # NodeWriters don't get scheduled
        if isinstance(writer, Writer):
            # Attach ScheduleWriter node to output
            if trigger is None:
                # Default to event trigger
                trigger = create_node(
                    "omni.graph.action.OnCustomEvent",
                    eventName=f"writerTrigger-{writer._writer_id}",
                    onlyPlayback=False,
                )
            elif isinstance(trigger, str):
                try:
                    trigger = create_node(trigger)
                except og.OmniGraphError as e:
                    raise WriterError(f"Invalid trigger. The name `{trigger}` is not a recognized node id.")
            elif isinstance(trigger, Callable):
                # Create on condition trigger
                trigger = on_condition(trigger)
            if isinstance(trigger, (ReplicatorItem, og.Node)):
                trigger_node = trigger.node if isinstance(trigger, ReplicatorItem) else trigger
                # print(f'DEBUG: WriterRegistry.attach() - 08')
                schedule_node = create_node("omni.replicator.core.OgnScheduleWriter", writer_id=writer._writer_id)
                # print(f'DEBUG: WriterRegistry.attach() - 09')
                ref_time_node = create_node("omni.replicator.core.ReadFabricTime")
                # print(f'DEBUG: WriterRegistry.attach() - 10')
                ref_time_node.get_attribute("outputs:fabricFrameTimeNumerator").connect(
                    schedule_node.get_attribute("inputs:rationalTimeOfSimNumerator"), True
                )
                ref_time_node.get_attribute("outputs:fabricFrameTimeDenominator").connect(
                    schedule_node.get_attribute("inputs:rationalTimeOfSimDenominator"), True
                )
                get_exec_attr(trigger_node, on_input=False).connect(schedule_node.get_attribute("inputs:exec"), True)
                # print(f'DEBUG: WriterRegistry.attach() - 11')
            elif trigger is not None:
                raise ValueError(f"Invalid trigger specified of type `{type(trigger)}`.")

        # Send writer attached event
        message_bus = omni.kit.app.get_app().get_message_bus_event_stream()
        message_bus.dispatch(
            carb.events.type_from_string("omni.replicator.core.writers"), payload={"attached": writer._writer_id}
        )
        # print(f'DEBUG: WriterRegistry.attach() - 12')

    @classmethod
    def _attach(
        cls, writer, writer_id, render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]], **kwargs
    ) -> None:
        writer_name = writer.__class__.__name__
        controller = og.Controller()
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise WriterRegistryError

        # print(f'DEBUG: WriterRegistry._attach() - 00')
        if stage.GetPrimAtPath(GRAPH_PATH):
            graph = controller.graph(GRAPH_PATH)
        else:
            SyntheticData.Get().activate_node_template("DispatchSync")
            # print(f'DEBUG: WriterRegistry._attach() - 01')
            graph = controller.graph(GRAPH_PATH)

        # If multiple render products
        if len(render_products) > 1:
            writer_node_name = f"MultipleRenderProducts_{len(render_products)}Products_{writer_name}Writer"
        else:
            writer_node_name = f"{render_products[0].split('/')[-1]}_{writer_name}Writer"

        # print(f'DEBUG: WriterRegistry._attach() - 02')
        writer_node_path = omni.usd.get_stage_next_free_path(stage, f"{GRAPH_PATH}/{writer_node_name}", False)
        # print(f'DEBUG: WriterRegistry._attach() - 03')
        if isinstance(writer, NodeWriter):
            # print(f'DEBUG: WriterRegistry._attach() - 04')
            writer_node = controller.create_node(writer_node_path, writer.node_type_id)
            writer_exec = get_exec_attr(writer_node, on_input=True)
            # print(f'DEBUG: WriterRegistry._attach() - 05')
            _create_node_attribute(writer_node, "inputs:writerId", og.Type(og.BaseDataType.TOKEN, 1, 0))
            # print(f'DEBUG: WriterRegistry._attach() - 06')
            og.AttributeValueHelper(writer_node.get_attribute("inputs:writerId")).set(writer_id, update_usd=True)
            if writer_exec is None:
                raise InvalidWriterError(
                    "Writer node `{writer.node_type_id}` is invalid, requires an input execution attribute."
                )
            writer_exec_in = writer_exec.get_name()
        else:
            # print(f'DEBUG: WriterRegistry._attach() - 07')
            writer_node = controller.create_node(writer_node_path, "omni.replicator.core.OgnWriter")
            # print(f'DEBUG: WriterRegistry._attach() - 08')
            og.AttributeValueHelper(writer_node.get_attribute("inputs:writerId")).set(writer_id, update_usd=True)
            og.AttributeValueHelper(writer_node.get_attribute("inputs:writerName")).set(writer_name, update_usd=True)
            og.AttributeValueHelper(writer_node.get_attribute("inputs:renderProducts")).set(
                render_products, update_usd=True
            )
            writer_exec_in = "inputs:exec"

        # Set static writer parameters
        # print(f'DEBUG: WriterRegistry._attach() - 09')
        for attribute_name, value in kwargs.items():
            attribute_name = f"inputs:{attribute_name.replace('inputs:', '')}"
            if not writer_node.get_attribute_exists(attribute_name):
                raise WriterRegistryError(
                    f"Invalid attribute `{attribute_name}` provided does not exist in writer `{writer_name}`"
                )
            og.AttributeValueHelper(writer_node.get_attribute(attribute_name)).set(value, update_usd=True)

        # print(f'DEBUG: WriterRegistry._attach() - 10')
        # Create SyncGate and trigger gate, attach to OgnWriter
        sync_gate_path = omni.usd.get_stage_next_free_path(stage, f"{GRAPH_PATH}/WriterSyncGate", False)
        # print(f'DEBUG: WriterRegistry._attach() - 11')
        sync_gate_node = controller.create_node(sync_gate_path, "omni.graph.action.RationalTimeSyncGate")
        # print(f'DEBUG: WriterRegistry._attach() - 12')
        _connect_attributes(
            sync_gate_node, writer_node, ["outputs:rationalTimeNumerator"], ["inputs:referenceTimeNumerator"]
        )
        _connect_attributes(
            sync_gate_node, writer_node, ["outputs:rationalTimeDenominator"], ["inputs:referenceTimeDenominator"]
        )
        _connect_attributes(sync_gate_node, writer_node, ["outputs:execOut"], [writer_exec_in])
        # trigger_gate_path = omni.usd.get_stage_next_free_path(stage, f"{GRAPH_PATH}/WriterTriggerGate", False)
        # trigger_gate_node = controller.create_node(trigger_gate_path, "omni.replicator.core.OgnTriggerGate")
        # _connect_attributes(sync_gate_node, trigger_gate_node, ["outputs:execOut"], ["inputs:exec"])
        # _connect_attributes(trigger_gate_node, writer_node, ["outputs:exec"], ["inputs:exec"])

        # if trigger:
        #     _attach_trigger(trigger, [trigger_gate_node], "inputs:triggerExec")

        # Add resolution and cameras from render products as Writer node attributes
        _create_node_attribute(writer_node, "inputs:render_products:resolution", og.Type(og.BaseDataType.INT, 2, 1))
        _create_node_attribute(writer_node, "inputs:render_products:name", og.Type(og.BaseDataType.TOKEN, 1, 1))
        _create_node_attribute(writer_node, "inputs:render_products:camera", og.Type(og.BaseDataType.TOKEN, 1, 1))

        names = []
        resolutions = []
        cameras = []

        for render_product in render_products:
            # print(f'DEBUG: WriterRegistry._attach() - 13')
            names.append(render_product.split("/")[-1])
            # print(f'DEBUG: WriterRegistry._attach() - 14')
            render_product_prim = stage.GetPrimAtPath(render_product)
            resolutions.append(render_product_prim.GetAttribute("resolution").Get())
            camera = render_product_prim.GetRelationship("camera").GetTargets()
            # print(f'DEBUG: WriterRegistry._attach() - 15')
            camera = str(camera[0]) if len(camera) else ""
            # print(f'DEBUG: WriterRegistry._attach() - 16')
            cameras.append(camera)

        og.AttributeValueHelper(writer_node.get_attribute("inputs:render_products:resolution")).set(
            resolutions, update_usd=True
        )
        og.AttributeValueHelper(writer_node.get_attribute("inputs:render_products:name")).set(names, update_usd=True)
        og.AttributeValueHelper(writer_node.get_attribute("inputs:render_products:camera")).set(
            cameras, update_usd=True
        )

        # Create and/or connect annotators to writer
        # print(f'DEBUG: WriterRegistry._attach() - 17')
        for annotator in writer.annotators:
            if isinstance(annotator, str):
                # print(f'DEBUG: WriterRegistry._attach() - 000')
                annotator = AnnotatorRegistry.get_annotator(annotator)

            if not isinstance(annotator, (Annotator, SyntheticData.NodeConnectionTemplate)):
                msg = f"The annotator specified by writer {writer_name} is not registered: {annotator}"
                raise InvalidWriterError(writer_name, writer, msg)

            if isinstance(writer, NodeWriter):
                # print(f'DEBUG: WriterRegistry._attach() - 001')
                if isinstance(annotator, SyntheticData.NodeConnectionTemplate):
                    attributes_mapping = annotator.attributes_mapping
                    # print(f'DEBUG: WriterRegistry._attach() - 18')
                    annotator = Annotator(
                        annotator.node_template_id,
                        render_product_idxs=annotator.render_product_idxs,
                        template_name=annotator.node_template_id,
                    )
                    # print(f'DEBUG: WriterRegistry._attach() - 19')
                    annotator.attach(render_products)

                    # Connect
                    for a_up, a_dwn in attributes_mapping.items():
                        if annotator.get_node().get_attribute_exists(a_up) and writer_node.get_attribute_exists(a_dwn):
                            attr_upstream = annotator.get_node().get_attribute(a_up)
                            if attr_upstream.get_resolved_type().get_role_name() == "execution":
                                attr_upstream.connect(sync_gate_node.get_attribute("inputs:execIn"))
                            else:
                                attr_upstream.connect(writer_node.get_attribute(a_dwn), True)
                    # print(f'DEBUG: WriterRegistry._attach() - 20')
                elif isinstance(annotator, Annotator):
                    # print(f'DEBUG: WriterRegistry._attach() - 21')
                    annotator.attach([render_product])
                    # print(f'DEBUG: WriterRegistry._attach() - 22')

                # print(f'DEBUG: WriterRegistry._attach() - 002')
                auto_connect(annotator.get_node(), writer_node, no_exec=True)
                _connect_execs(annotator.get_node(), sync_gate_node)
                # print(f'DEBUG: WriterRegistry._attach() - 23')
            else:
                # print(f'DEBUG: WriterRegistry._attach() - 24')
                if annotator._render_product_idxs is None:
                    annotator._render_product_idxs = [0]
                    for render_product in render_products:
                        annotator = annotator.attach([render_product])
                        _connect_to_writer(graph, sync_gate_node, writer_node, annotator)
                    # print(f'DEBUG: WriterRegistry._attach() - 25')
                else:
                    annotator = annotator.attach(render_products)
                    _connect_to_writer(graph, sync_gate_node, writer_node, annotator)
                    # print(f'DEBUG: WriterRegistry._attach() - 26')

        # Connect trigger to dispatch gate
        dispatcher_node_path = f"{GRAPH_PATH}/PostProcessDispatcher"
        try:
            # print(f'DEBUG: WriterRegistry._attach() - 003')
            dispatcher_node = controller.node(dispatcher_node_path)
            # print(f'DEBUG: WriterRegistry._attach() - 004')
        except og.OmniGraphValueError:
            # print(f'DEBUG: WriterRegistry._attach() - 005')
            SyntheticData.Get().activate_node_template("DispatchSync")
            dispatcher_node = controller.node(dispatcher_node_path)
            # print(f'DEBUG: WriterRegistry._attach() - 006')

        # print(f'DEBUG: dispatcher_node.get_attribute("outputs:exec").get_downstream_connections() = {dispatcher_node.get_attribute("outputs:exec").get_downstream_connections()}')
        connections = dispatcher_node.get_attribute("outputs:exec").get_downstream_connections()
        if len(connections):
            dispatch_sync_gate_node = (
                dispatcher_node.get_attribute("outputs:exec").get_downstream_connections()[0].get_node()
            )
        else:
            dispatch_sync_gate_node = None

        if dispatch_sync_gate_node:
            # print(f'DEBUG: dispatch_sync_gate_node.get_attribute("outputs:exec").get_downstream_connections() = {dispatch_sync_gate_node.get_attribute("outputs:exec").get_downstream_connections()}')
            connections = dispatch_sync_gate_node.get_attribute("outputs:exec").get_downstream_connections()
            if len(connections):
                dispatch_gate_node = (
                    dispatch_sync_gate_node.get_attribute("outputs:exec").get_downstream_connections()[0].get_node()
                )
        # print(f'DEBUG: WriterRegistry._attach() - 27')

        # Attach trigger to dispatch trigger gate
        # if trigger:
        #     _attach_trigger(trigger, [dispatch_gate_node], exec_in_name="inputs:triggerExec")

        # Connect time to SyncGate rational
        _connect_attributes(
            dispatcher_node, sync_gate_node, ["outputs:referenceTimeNumerator"], ["inputs:rationalTimeNumerator"]
        )
        _connect_attributes(
            dispatcher_node, sync_gate_node, ["outputs:referenceTimeDenominator"], ["inputs:rationalTimeDenominator"]
        )
        # _connect_attributes(dispatcher_node, trigger_gate_node, ["outputs:referenceTimeNumerator"], ["inputs:rationalTimeNumerator"])
        # _connect_attributes(dispatcher_node, trigger_gate_node, ["outputs:referenceTimeDenominator"], ["inputs:rationalTimeDenominator"])

        # Enable frame gate
        frame_gate_node_path = f"{GRAPH_PATH}/DispatchSync"
        frame_gate_node = controller.node(frame_gate_node_path)
        frame_gate_node.get_attribute("inputs:enabled").set(True)

        # print(f'DEBUG: WriterRegistry._attach() - 28')
        return writer_node

    @classmethod
    def detach(
        cls, writer_name: str, render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]] = None
    ) -> None:
        """
        Detach active writer with specified id from render_products.

        Args:
            writer_name: Name of writer(s) to be detached.
            render_products: List of render product prim paths from which to disable the
                specified writer. If not provided, writers of matching `writer_name` will be disabled
                from all render products.
        """
        writers_to_detach = []
        for render_products in cls._active_writers:
            for writer_id, cur_writer in cls._active_writers[render_products].items():
                if cur_writer.__class__.__name__ == writer_name:
                    writers_to_detach.append(writer_id)

        for writer_id in writers_to_detach:
            cls._detach_by_writer_id(writer_id)

    @classmethod
    def _detach_by_writer_id(cls, writer_id: str) -> None:
        """
        Detach active writer with specified id from render_products.

        Args:
            writer_id: Id of activated writer.
        """
        render_products_to_remove = []
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise WriterRegistryError("Invalid USD stage, unable to attach writer")
        else:
            session_layer = stage.GetSessionLayer()
            with Usd.EditContext(stage, session_layer):
                for render_products in cls._active_writers:
                    if writer_id in cls._active_writers[render_products]:
                        # Destroy sync gate and writer
                        writer_node = cls._get_attached_writer_node(writer_id)
                        if writer_node:
                            graph = writer_node.get_graph()
                            writer_exec = get_exec_attr(writer_node, on_input=True)
                            sync_gate = writer_exec.get_upstream_connections()[0].get_node()
                            graph.destroy_node(writer_node.get_prim_path(), True)
                            graph.destroy_node(sync_gate.get_prim_path(), True)

                        writer = cls._active_writers[render_products].pop(writer_id)

                        # Deactivate Annotators
                        for annotator in writer.annotators:
                            AnnotatorRegistry.detach(annotator, render_products)

                        if not cls._active_writers[render_products]:
                            # Create a list of empty render products to clean up
                            render_products_to_remove.append(render_products)

                        # if only dispatchers and syncs left, remove them
                        remaining_nodes = graph.get_nodes()
                        non_annotators = ["omni.syntheticdata.SdOnNewFrame", "omni.replicator.core.OgnRefTimeGate"]
                        if all([rn.get_type_name() in non_annotators for rn in remaining_nodes]):
                            for rn in remaining_nodes:
                                graph.destroy_node(rn.get_prim_path(), True)
                            SyntheticData.Get().reset()  # Clear activation history

            # Destroy writer scheduler nodes tied to writer
            scheduler_graph = og.Controller().graph("/Replicator/SDGPipeline")
            for node in scheduler_graph.get_nodes():
                if (
                    node.get_type_name() == "omni.replicator.core.OgnScheduleWriter"
                    and node.get_attribute("inputs:writer_id").get() == writer_id
                ):
                    fabric_read_node = (
                        node.get_attribute("inputs:rationalTimeOfSimDenominator")
                        .get_upstream_connections()[0]
                        .get_node()
                    )
                    writer_trigger = node.get_attribute("inputs:exec").get_upstream_connections()[0].get_node()

                    scheduler_graph.destroy_node(fabric_read_node.get_prim_path(), True)
                    scheduler_graph.destroy_node(writer_trigger.get_prim_path(), True)
                    scheduler_graph.destroy_node(node.get_prim_path(), True)
                    break

        # Clean up render_products if they are not associated with any writer
        for render_products in render_products_to_remove:
            cls._active_writers.pop(render_products)

    @classmethod
    def _detach_by_writer(cls, writer: Writer) -> None:
        """
        Detach active writer with specified id from render_products.

        Args:
            writer: Writer instance
        """
        writers_to_detach = []
        for render_products in cls._active_writers:
            for writer_id, cur_writer in cls._active_writers[render_products].items():
                if cur_writer == writer:
                    writers_to_detach.append(writer_id)

        for writer_id in writers_to_detach:
            cls._detach_by_writer_id(writer_id)

    @classmethod
    def get_writers(cls, category: str = None) -> dict:
        """
        Return dictionary of registered writers with mapping `{writer_name: writer}`.

        Args:
            category: Optionally specify the category of the writers to retrieve.
        """
        if category:
            writer_names = cls._categories.get(category, [])
            return {name: cls._writers[name] for name in writer_names}
        else:
            return dict(cls._writers)

    @classmethod
    def get(
        cls,
        writer_name: str,
        init_params: Dict = None,
        render_products: List[Union[str, HydraTexture]] = None,
        trigger: Union[ReplicatorItem, Callable] = "omni.replicator.core.OgnOnFrame",
    ) -> Writer:
        """Get a registered writer

        Args:
            writer_name: Writer name
            init_params: Dictionary of initialization parameters with which to initialize writer
            render_products: List of render products to attach to writer
            trigger: Function or replicator trigger that triggers the `write` function of the writer. If a function
                is supplied, it must return a boolean. If set to `None`, the writer is set to a manual mode
                where it can be triggered by calling `writer.schedule_write`.
        """
        if writer_name not in cls._writers:
            raise WriterRegistryError(f"No writer with name `{writer_name}` was found in registry.")
        if isinstance(cls._writers[writer_name], NodeWriter):
            writer = cls._writers[writer_name]
        elif issubclass(cls._writers[writer_name], Writer):
            writer = cls._writers[writer_name].__new__(cls._writers[writer_name])
            # writer = cls._writers[writer_name]
        else:
            raise WriterRegistryError(f"No writer named `{writer_name}` found in registry.")
        if init_params:
            writer.initialize(**init_params)
        if render_products:
            writer.attach(render_products, trigger=trigger)
        return writer

    @classmethod
    def _on_final_frame(cls):
        for _, writers in cls._active_writers.items():
            for _, writer in writers.items():
                writer.on_final_frame()

    @classmethod
    def _get_attached_writer_node(cls, writer_id):
        stage = omni.usd.get_context().get_stage()
        for node_prim in stage.GetPrimAtPath(GRAPH_PATH).GetChildren():
            # Skip non-nodes
            try:
                node = og.Controller().node(node_prim)
            except og.OmniGraphValueError:
                continue
            if (
                node.get_attribute_exists("inputs:writerId")
                and node.get_attribute("inputs:writerId").get() == writer_id
            ):
                return node

    @classmethod
    def get_attached_writers(
        cls, render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]] = None
    ) -> dict:
        """
        Return dictionary of enabled writers with mapping `{writer_name: writer}`.

        Args:
            render_products ([string, list[string]], optional): Render Product prim path(s).
        """
        if render_products is None:
            raise WriterRegistryError("Render products must be defined")
        else:
            writers = {}
            enabled_writers = set()
            if isinstance(render_products, (str, HydraTexture)):
                render_products = [render_products]
            render_products = [rp.path if isinstance(rp, HydraTexture) else rp for rp in render_products]
            for render_product in render_products:
                enabled_writers = enabled_writers.union(cls._active_writers.get((render_product,), {}))

                for writer_id in enabled_writers:
                    writer_name = cls._active_writers[(render_product,)][writer_id].__class__.__name__
                    if writer_name not in writers:
                        writers[writer_name] = cls._active_writers[(render_product,)][writer_id]
                    else:
                        carb.log_warn(
                            "Deprecated: Multiple writers of the same type not supported with this function. Returning first instance"
                        )

        return writers

    @classmethod
    def get_attached_writer(
        cls, writer_name: str, render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]]
    ) -> Writer:
        """
        Return the Writer object of `writer_name` on `render_products`.

        Args:
            writer_name: Writer name
            render_products: Render Product prim path(s)
        """
        if isinstance(render_products, (str, HydraTexture)):
            render_products = [render_products]
        render_products = [rp.path if isinstance(rp, HydraTexture) else rp for rp in render_products]

        writer = None
        for render_product in render_products:
            if (render_product,) not in cls._active_writers:
                raise WriterRegistryError(
                    f"No writer named {writer_name} found attached to the render products {render_products}"
                )
            else:
                # Set writer to the first match and break
                for writer_id in cls._active_writers[(render_product,)]:
                    if (
                        cls._active_writers[(render_product,)][writer_id].__class__.__name__ == writer_name
                        and not writer
                    ):
                        writer = cls._active_writers[(render_product,)][writer_id]
                    else:
                        carb.log_warn(
                            f"Deprecated: Multiple {writer_name} writers on {render_product}. Returning the first {writer_name} writer on the render product {render_product}."
                        )
                        break

        return writer

    @classmethod
    def _get_attached_writer(cls, writer_id: str = None) -> Writer:
        if writer_id is None:
            raise WriterRegistryError("Writer ID must be defined.")

        writer = None
        for render_product in cls._active_writers:
            if writer_id in cls._active_writers[render_product]:
                writer = cls._active_writers[render_product][writer_id]
                break

        if writer:
            return writer
        else:
            raise WriterRegistryError(f"No writer with ID {writer_id} found attached to the render products.")

    @classmethod
    def get_writer_render_products(cls) -> set:
        """
        Return set of all render_products that are associated with one or more enabled writers.
        """
        return set(cls._active_writers.keys())

    @classmethod
    def get_annotators(cls, render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]] = None) -> set:
        """
        Return a set of annotators required by all enabled writers associated with `render_products`.

        Args:
            render_products: Render Product prim path(s).
        """
        return {
            annotator
            for writer in cls.get_attached_writers(render_products).values()
            for annotator in writer.annotators
        }

    @classmethod
    async def initialize_annotators(
        cls, render_products: Union[str, HydraTexture, List[Union[str, HydraTexture]]] = None
    ) -> None:
        """
        Call the `initialize_fn` of the annotators associated with `render_products`, if available.

        Args:
            render_products: Render Product prim path(s).
        """
        vp_iface = omni.kit.viewport.get_viewport_interface()
        viewport_instances = vp_iface.get_instance_list()

        # Get render_product to viewport mapping
        render_product_vp_mapping = {}
        for vpi in viewport_instances:
            vp = vp_iface.get_viewport_window(vpi)
            render_product_vp_mapping[vp.get_active_render_product()] = vp

        if render_products is None:
            render_products = list(cls.get_writer_render_products())
        elif isinstance(render_products, (str, HydraTexture)):
            render_products = [render_products]

        for render_product in render_products:
            vp = render_product_vp_mapping.get(render_product)
            if not vp:
                raise WriterRegistryError(f"No viewport attached to render_product `{render_product}`")
            for annotator in cls.get_annotators(render_product):
                init_fn = AnnotatorRegistry.get_initialize_fn(annotator)
                if AnnotatorRegistry.get_is_viewport_annotator(annotator):
                    args = [vp]
                else:
                    args = []

                if inspect.iscoroutinefunction(init_fn):
                    await init_fn(*args)
                elif isinstance(init_fn, partial) and inspect.iscoroutinefunction(init_fn.func):
                    await init_fn(*args)
                else:
                    init_fn(*args)

    @classmethod
    def detach_all(cls) -> None:
        """Detach all active writers"""
        writers_to_detach = []
        for writer_dict in cls._active_writers.values():
            for writer in writer_dict.values():
                writers_to_detach.append(writer)

        for writer in writers_to_detach:
            cls._detach_by_writer(writer)

    @classmethod
    def _reset(cls):
        """Clears stored active writers

        Clears active writer but does not affect the graph."""
        cls._active_writers.clear()


def get(
    name: str,
    init_params: Dict = None,
    render_products: List[Union[str, HydraTexture]] = None,
    trigger: Union[ReplicatorItem, Callable] = "omni.replicator.core.OgnOnFrame",
) -> Writer:
    """Get writer instance from registered writers

    Args:
        name: Writer name
        init_params: Dictionary of initialization parameters with which to initialize writer
        render_products: List of render products to attach to writer
        trigger: Function or replicator trigger that triggers the `write` function of the writer. If a function
                is supplied, it must return a boolean. If set to `None`, the writer is set to a manual mode
                where it can be triggered by calling `writer.schedule_write`.

    Example:
        >>> import omni.replicator.core as rep
        >>> rp = rep.create.render_product(rep.create.camera(), (512, 512))
        >>> writer = rep.writers.get(
        ...     name="BasicWriter",
        ...     init_params={"output_dir": "_out", "rgb": True},
        ...     render_products=rp,
        ...     trigger=None,
        ... )
    """
    return WriterRegistry.get(name, init_params, render_products, trigger)


def unregister_writer(writer_name) -> None:
    """
    Unregister a writer with specified name if it exists.

    Args:
        writer_name: Name of registered writer.

    Example:
        >>> import omni.replicator.core as rep
        >>> class MyWriter(rep.Writer):
        ...     def __init__(self):
        ...         self.annotators = ["LdrColor"]
        ...         self.frame_id
        ...     def write(self, data):
        ...         print(f"I have data #{self.frame_id}!")
        ...         self.frame_id += 1
        >>> rep.writers.register_writer(MyWriter)
        >>> rep.writers.unregister_writer("MyWriter")
    """
    WriterRegistry.unregister(writer_name)


def register_writer(writer: Writer, category: str = None) -> None:
    """Register a writer.

    Registered writers can be retrieved with `WriterRegistry.get(<writer_name>)`

    Args:
        writer: Instance of class `Writer`.
        category: Optionally specify a category of writer to group writers together.

    Example:
        >>> import omni.replicator.core as rep
        >>> class MyWriter(rep.Writer):
        ...     def __init__(self):
        ...         self.annotators = ["LdrColor"]
        ...         self.frame_id
        ...     def write(self, data):
        ...         print(f"I have data #{self.frame_id}!")
        ...         self.frame_id += 1
        >>> rep.writers.register_writer(MyWriter)
    """
    WriterRegistry.register(writer, category)


def register_node_writer(name: str, node_type_id: str, annotators: List, category: str = None, **kwargs) -> None:
    """Register a Node Writer

    Register a writer implemented as an omnigraph node.

    Args:
        node_type_id: The node's type identifier (eg. `'my.extension.OgnCustomNode'`)
        annotators: List of dependent annotators
        category: Optionally specify a category of writer to group writers together.
        kwargs: Node Writer input attribute initialization
    """
    WriterRegistry.register_node_writer(
        name=name, annotators=annotators, node_type_id=node_type_id, category=category, **kwargs
    )
