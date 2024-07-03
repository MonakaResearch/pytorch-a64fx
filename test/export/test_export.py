# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import dataclasses
import io
import logging
import re
import unittest
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from re import escape
from typing import Dict, List

import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F

from functorch.experimental.control_flow import cond, map
from torch import Tensor
from torch._dynamo.test_case import TestCase
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
from torch._subclasses import FakeTensorMode
from torch.export import Dim, dynamic_dim, export, unflatten
from torch.export._trace import (
    _export,
    _export_to_torch_ir,
    DEFAULT_EXPORT_DYNAMO_CONFIG,
)
from torch.export.graph_signature import InputKind
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import onlyCPU, onlyCUDA
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    TEST_TRANSFORMERS,
    TestCase as TorchTestCase,
)
from torch.utils._pytree import (
    LeafSpec,
    tree_flatten,
    tree_map,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_loads,
)

try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    HAS_TORCHREC = True
except ImportError:
    HAS_TORCHREC = False

try:
    from . import testing
except ImportError:
    import testing
# The following import pattern matters as `test_export.export` is patched
# in other files (like test_export_nonstrict.py). `torch.export.export`
# will invalidate the patch.
from torch.export import export


torch.library.define("testlib::returns_tensor_symint", "(Tensor x) -> (Tensor, SymInt)")
torch.library.define(
    "testlib::foo",
    "(Tensor(a!) x, Tensor(b!) z) -> (Tensor, Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_mutated",
    "(Tensor(a!) x) -> (Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_functional",
    "(Tensor x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_unbacked",
    "(Scalar x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("testlib::returns_tensor_symint", "cpu")
@torch.library.impl_abstract("testlib::returns_tensor_symint")
def returns_tensor_symint_impl(x):
    return x, x.shape[0]


@torch.library.impl("testlib::foo", "cpu")
@torch._dynamo.disable
def foo_impl(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z


@torch.library.impl_abstract("testlib::foo")
def foo_abstract(x, z):
    return x, z, x + z


@torch.library.impl("testlib::foo_mutated", "CompositeImplicitAutograd")
def foo_mutated(x):
    a, b, c = torch.ops.testlib.foo(x, x.cos())
    return a, a.cos()


@torch.library.impl("testlib::foo_functional", "CompositeImplicitAutograd")
def foo_functional(x):
    a, b, c = torch.ops.testlib.foo(x.cos(), x.cos())
    return a.cos()


@torch.library.impl("testlib::foo_unbacked", "CompositeImplicitAutograd")
def foo_unbacked(x):
    if x > 2:
        return torch.ones(4, 4)
    if x < 6:
        return torch.ones(4, 4)
    return torch.ones(4, 4)


@dataclass
class Inp:
    x: Tensor
    y: List[Tensor]
    z: Dict[str, Tensor]


NON_STRICT_SUFFIX = "_non_strict"
RETRACEABILITY_SUFFIX = "_retraceability"
SERDES_SUFFIX = "_serdes"
PREDISPATCH_SUFFIX = "_pre_dispatch"
TRAINING_IR_DECOMP_SUFFIX = "_training_ir_to_decomp"


def is_non_strict_test(test_name):
    return test_name.endswith(NON_STRICT_SUFFIX)


def is_retracebility_test(test_name):
    return test_name.endswith(RETRACEABILITY_SUFFIX)


def is_serdes_test(test_name):
    return test_name.endswith(SERDES_SUFFIX)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    def test_export_inline_constraints(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                torch._check_is_size(b)
                return torch.full((b, 1), 1)

        f = Module()
        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = export(f, inp)
        res = gm.module()(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    def test_export_constraints_error_not_in_range(self):
        class InvalidInputConflictWithInputConstraints(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.zeros([3])
        dim_x = torch.export.Dim("dim_x", min=6)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "not in range"):
            torch.export.export(
                InvalidInputConflictWithInputConstraints(),
                (inp,),
                dynamic_shapes={"x": {0: dim_x}},
            )

    def test_export_slice_maxsize(self):
        class Slice(torch.nn.Module):
            def forward(self, *args):
                return torch.ops.aten.slice.Tensor(*args)

        inp = (torch.rand((10, 3, 224, 224)), 0, 0, 9223372036854775807)
        dynamic_shapes = (({0: Dim("dim")}, None, None, None),)
        torch.export.export(
            Slice(),
            inp,
            dynamic_shapes=dynamic_shapes,
        )

    def test_export_constraints_error(self):
        class ConflictingConstraints(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                torch._check_is_size(b)
                torch._check(b >= 4)
                torch._check(b <= 5)
                return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ep = export(ConflictingConstraints(), inp)

        with self.assertRaisesRegex(
            RuntimeError, r"Invalid value range for 3 between \[4, 5\]"
        ):
            ep.module()(torch.tensor([3]))

    def test_export_assume_static_by_default(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                if x.shape[0] == 4:
                    return x + 1
                else:
                    return x

        branch_on_shape = Module()
        inp = (torch.rand(4, 5),)

        # Being able to export means shape is preserved as static
        export(branch_on_shape, inp)


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):
    def _test_export_same_as_eager(self, f, args, kwargs=None):
        kwargs = kwargs or {}
        exported_program = export(f, args, kwargs)
        self.assertEqual(exported_program.module()(*args, **kwargs), f(*args, **kwargs))
        # this is not supported by .module()
        # reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        # self.assertEqual(
        #     exported_program.module()(*args, **reversed_kwargs), f(*args, **reversed_kwargs)
        # )

    def test_basic(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x[0] + y

        f = Module()
        inp = ([torch.ones(1, 3)], torch.ones(1, 3))
        self._test_export_same_as_eager(f, inp)

    # Errors because non-strict is not supported in training IR (T193692164)
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_external_call_non_strict_real_tensor(self):
        class ExternalMethod:
            def add(self, x):
                return x + x

        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.external_add = ExternalMethod().add

            def forward(self, x):
                return self.external_add(x)

        f = Basic()
        args = (torch.randn(1, 3),)
        ep = export(f, args, strict=False)
        self.assertEqual(ep.module()(*args), f(*args))

    def test_colon_parameter(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("foo:bar", torch.nn.Parameter(torch.ones(3, 3)))

            def forward(self, x):
                return x + getattr(self, "foo:bar")

        ep = export(M(), (torch.randn(3, 3),))
        x = torch.randn(3, 3)
        self.assertEqual(ep.module()(x), M()(x))

    def test_conv_dynamic(self):
        # Simple module for demonstration
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                a = self.conv(x)
                a.add_(y)
                return self.maxpool(self.relu(a))

        example_args = (torch.randn(2, 3, 256, 256), torch.ones(2, 32, 256, 256))
        dynamic_shapes = {"x": {0: Dim("batch")}, "y": {0: Dim("batch")}}
        m = M()
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        self.assertEqual(exported_program.module()(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(exported_program.module()(*args), m(*args))

        from torch._export import capture_pre_autograd_graph

        gm: torch.fx.GraphModule = capture_pre_autograd_graph(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        self.assertEqual(gm(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(gm(*args), m(*args))

    # Errors because non-strict is not supported in training IR (T193692164)
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_basic_non_strict_real_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(1, 3))

            def forward(self, x, y):
                return x[0] + y - self.param

        f = Basic()
        args = ([torch.randn(1, 3)], torch.randn(1, 3))
        ep = export(f, args, strict=False)
        self.assertEqual(ep.module()(*args), f(*args))

    # Errors because non-strict is not supported in training IR (T193692164)
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_basic_non_strict_fake_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x, y):
                return x[0] + y - self.param

        fake_mode = FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))
        f = Basic()
        with fake_mode:
            args = ([torch.empty(3, 2)], torch.empty(3, 2))
        ep = export(f, args, strict=False)
        inputs = ([torch.randn(3, 2)], torch.randn(3, 2))
        self.assertEqual(ep.module()(*inputs), f(*inputs))

    def test_non_strict_dynamic_shapes(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("u", torch.ones(1))
                self.register_buffer("v", torch.ones(1))

            def forward(self, x, ys, zs, c):
                y = ys[0] + ys[1] + zs["a"] + zs["b"]
                self.v.add_(3)
                w = self.u - self.v
                if x.shape[0] < 3 and c.shape[0] != 4:
                    return x + w, x + y
                else:
                    return x - w, x - y

        foo = Foo()

        inp = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            [{0: dim}, {0: dim}],
            {"a": {0: dim}, "b": {0: dim}},
            None,
        )

        ep_ns = torch.export.export(
            foo, inp, dynamic_shapes=dynamic_shapes, strict=False
        )

        bad_runtime_inp1 = (
            torch.ones(6),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape(
                "Expected input at *args[1][0].shape[0] to be equal to 6, but got 5"
            ),
        ):
            ep_ns.module()(*bad_runtime_inp1)

        bad_runtime_inp2 = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(6),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[3].shape[0] to be equal to 4, but got 6"),
        ):
            ep_ns.module()(*bad_runtime_inp2)

        good_runtime_inp = (
            torch.ones(7),
            [torch.zeros(7), torch.ones(7)],
            {"a": torch.zeros(7), "b": torch.ones(7)},
            torch.ones(4),
        )
        ep_ns.module()(*good_runtime_inp)

        bad_example_inp = (
            torch.ones(2),
            [torch.zeros(2), torch.ones(2)],
            {"a": torch.zeros(2), "b": torch.ones(2)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            "2 not in range.*3,",
        ):
            ep_ns = torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_non_strict_dynamic_shapes_suggested_fixes(self):
        class Foo(torch.nn.Module):
            def forward(self, x, c):
                if x.shape[0] <= 6:
                    return x + 1, c + 2
                else:
                    return x - 1, c - 2

        foo = Foo()

        bad_example_inp = (
            torch.ones(5),
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            None,
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated \\(dim\\)!(.*\n)*.*"
            "Not all values of dim.*satisfy the generated guard(.*\n)*.*"
            "Suggested fixes:(.*\n)*.*"
            "dim = Dim\\('dim', min=3, max=6\\)",
        ):
            torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_state_tensors(self):
        class M(torch.nn.Module):  # simple with register buffer
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(2, 3), persistent=False)

            def forward(self, x):
                # x = 2
                y = self.buf
                # y = 1
                w1 = self.buf + 3
                w2 = self.buf + 4
                w3 = self.buf + 5
                self.buf = w1
                z = self.buf
                self.buf = w3
                # z = 4
                return x + y + z + w2

        ep = torch.export.export(M(), (torch.randn(2, 3),), strict=False)
        self.assertEqual(ep.graph_signature.buffers_to_mutate, {"add_2": "buf"})
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(2, 3) + 1), torch.ones(2, 3) * 12)
        )

        class M(torch.nn.Module):  # simple without register buffer
            def __init__(self):
                super().__init__()
                self.buf = torch.ones(2, 3)

            def forward(self, x):
                # x = 2
                y = self.buf
                # y = 1
                self.buf = self.buf + 3
                z = self.buf
                # z = 3
                return x + y + z

        with self.assertRaisesRegex(
            ValueError,
            "The tensor attribute self.buf was assigned during export",
        ):
            torch.export.export(M(), (torch.randn(2, 3),), strict=False)

        class M(torch.nn.Module):  # complex with register buffer
            def __init__(self):
                super().__init__()
                tensors = [torch.ones(2, 3), torch.ones(2, 3)]
                for i, tensor in enumerate(tensors):
                    self.register_buffer(f"buf_{i}", tensor, persistent=False)

            def get_tensor(self, i):
                return getattr(self, f"buf_{i}")

            def set_tensor(self, i, val):
                setattr(self, f"buf_{i}", val)

            def forward(self, x):
                # x = 2
                y = self.get_tensor(0) + self.get_tensor(1)
                # y = 1 + 1
                self.set_tensor(0, torch.ones(2, 3) + 2)
                self.set_tensor(1, torch.ones(2, 3) + 2)
                z = self.get_tensor(0) + self.get_tensor(1)
                # z = 3 + 3
                return x + y + z

        ep = torch.export.export(M(), (torch.randn(2, 3),), strict=False)
        self.assertEqual(
            ep.graph_signature.buffers_to_mutate, {"add_1": "buf_0", "add_2": "buf_1"}
        )
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(2, 3) + 1), torch.ones(2, 3) * 10)
        )

        class M(torch.nn.Module):  # complex without register buffer
            def __init__(self):
                super().__init__()
                self.tensors = [torch.ones(2, 3), torch.ones(2, 3)]

            def get_tensor(self, i):
                return self.tensors[i]

            def set_tensor(self, i, val):
                self.tensors[i] = val

            def forward(self, x):
                # x = 2
                y = self.get_tensor(0) + self.get_tensor(1)
                # y = 1 + 1
                self.set_tensor(0, torch.ones(2, 3) + 2)
                self.set_tensor(1, torch.ones(2, 3) + 2)
                z = self.get_tensor(0) + self.get_tensor(1)
                # z = 3 + 3
                return x + y + z

        with self.assertRaisesRegex(
            ValueError,
            "The tensor attributes self.tensors\\[0\\], self.tensors\\[1\\] were assigned during export",
        ):
            torch.export.export(M(), (torch.randn(2, 3),), strict=False)

    def test_state_primitives(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = 1
                self.y = {"k": 2}
                self.z = (3,)

            def forward(self, x):
                self.x = self.x + 4
                self.y["k"] = self.y["k"] + 5
                self.z = (self.z[0] + 6,)
                return x + self.x + self.y["k"] + self.z[0]

        ep = export(M(), (torch.randn(2, 3),))
        self.assertTrue(
            torch.allclose(ep.module()(torch.zeros(2, 3)), torch.ones(2, 3) * 21)
        )

    # Predispatch has different expected results
    @testing.expectedFailureTrainingIRToRunDecomp  # T193700910
    def test_torch_fn(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.linear(x)
                x = self.relu(x)
                x = x + x
                return x

        ep1 = export(M1(), (torch.randn(3, 3),)).run_decompositions()
        expected_result = [
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_2", "builtin_function_or_method.linear"),
            ("linear_2", "builtin_function_or_method.linear"),
            ("relu_1", "function.relu"),
            ("add_1", "method_descriptor.add"),
        ]
        actual_result = []
        for i, node in enumerate(ep1.graph.nodes):
            if node.op == "call_function":
                actual_result.append(node.meta.get("torch_fn"))
        self.assertEqual(actual_result, expected_result)

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, bias):
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = torch.add(x, x)
                return x

        ep2 = export(
            M2(), (torch.randn(3, 3), torch.randn(3, 3), torch.randn(3))
        ).run_decompositions()
        expected_result = [
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_1", "builtin_function_or_method.linear"),
            ("relu_1", "function.relu"),
            ("add_1", "builtin_function_or_method.add"),
        ]
        actual_result = []
        for i, node in enumerate(ep2.graph.nodes):
            if node.op == "call_function":
                actual_result.append(node.meta.get("torch_fn"))
        self.assertEqual(actual_result, expected_result)

    def test_export_preserve_linear_at_aot_level(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return torch.ops.aten.chunk.default(x, 3, 0)

        gm = (
            torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            )
            .run_decompositions({}, _preserve_ops=(torch.ops.aten.linear.default,))
            .graph_module
        )
        # linear is CompositeImplicitAutograd functional op so we should preserve it
        # chunk is CompositeImplicitAutograd non-functional op we decompose.
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    split = torch.ops.aten.split.Tensor(linear, 1);  linear = None
    getitem = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2];  split = None
    return (getitem, getitem_1, getitem_2)""",
        )

    # TODO(yidi)
    # Expected failure for test cases that calls run_decomposition().
    # The top-level cond node has pre-existing metadata,
    # which overrides the metadata for operators in subgraph due to interpreter.run(),
    # where cond is a single node in the interpreter.run(). And we preserve metadata
    # by copying current node's metadata for all nodes created during interpreting.
    @testing.expectedFailurePreDispatchRunDecomp
    @testing.expectedFailureRetraceability
    @testing.expectedFailureTrainingIRToRunDecomp  # T193700910
    def test_export_cond_preserve_torch_fn_for_subgraphs(self):
        class MySubModule(torch.nn.Module):
            def foo(self, x):
                return x.cos()

            def forward(self, x):
                return self.foo(x)

        class CondBranchClassMethod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subm = MySubModule()

            def bar(self, x):
                return x.sin()

            def forward(self, x):
                return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])

        example_inputs = (torch.randn(1, 3, 3, 3),)
        m = CondBranchClassMethod()
        m.eval()
        gm = export(m, example_inputs).module()

        actual_torch_fns = []
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in {"sin", "cos"}:
                    torch_fn = node.meta.get("torch_fn")
                    print(torch_fn)
                    actual_torch_fns.append(torch_fn)
        exp_torch_fns = [
            ("cos_1", "method_descriptor.cos"),
            ("sin_1", "method_descriptor.sin"),
        ]
        self.assertEqual(actual_torch_fns, exp_torch_fns)

    def test_derived_dim_basic(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1:]

        foo = Foo()

        x, y = torch.randn(5), torch.randn(6)
        dimx = torch.export.Dim("dimx", min=3, max=6)

        dimy = torch.export.Dim("dimy", min=4, max=7)  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(dimy\\)!(.*\n)*.*"
                "The values of dimy.*must always be related to the values of dimx.*by.*(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dimy = dimx \\+ 1"
            ),
        ):
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        dimy = dimx * 2  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.* to be equal to 2\\*dimx, where dimx = 5, but got 6",
        ):
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        dimy = dimx + 1  # works
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 5, but got 6",
        ):
            ep.module()(torch.randn(4), torch.randn(6))

        self.assertEqual(ep.module()(torch.randn(4), torch.randn(5)).size()[0], 4)

    def test_derived_dim_nested(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1::2]

        foo = Foo()

        x, y = torch.randn(5), torch.randn(11)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimy = dimx * 2 + 1  # works
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        self.assertEqual(ep.module()(torch.randn(4), torch.randn(9)).size()[0], 4)

        class Foo(torch.nn.Module):
            def forward(self, z, y):
                return z[1:] + y[1::2]

        foo = Foo()

        z, y = torch.randn(6), torch.randn(11)

        dimz = dimx
        dimy = dimx * 2 - 1  # works
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )
        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)

        dimz = dimx + 1
        dimy = dimx * 2 - 1  # doesn't work

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.*to be equal to 2\\*dimx - 1, where dimx = 5, but got 11",
        ):
            export(
                foo,
                (z, y),
                dynamic_shapes=({0: dimz}, {0: dimy}),
            )

        dimy = dimx * 2 + 1  # works
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected input.*shape.*to be <= 7, but got 8"
        ):
            ep.module()(torch.randn(8), torch.randn(15))
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 9, but got 8",
        ):
            ep.module()(torch.randn(5), torch.randn(8))

        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)

    def test_derived_dim_integer(self):
        class Foo(torch.nn.Module):
            def forward(self, w):
                if w.shape[0] % 2 == 0:
                    return w[::2]
                else:
                    return w[1:-1:2]

        foo = Foo()

        w = torch.randn(10)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimw = dimx * 2 + 1  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected shape.*= 10 of input Tensor to be "
            "of the form 2\\*dimx \\+ 1, where dimx is an integer",
        ):
            export(
                foo,
                (w,),
                dynamic_shapes=({0: dimw},),
            )

        dimw = dimx * 2  # works
        ep = export(
            foo,
            (w,),
            dynamic_shapes=({0: dimw},),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*= 9 to be "
            "of the form 2\\*s1, where s1 is an integer",
        ):
            ep.module()(torch.randn(9))

        self.assertEqual(ep.module()(torch.randn(8)).size()[0], 4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be <= 12, but got 14",
        ):
            ep.module()(torch.randn(14))

    def test_derived_dim_repeat_derived(self):
        class Foo(torch.nn.Module):
            def forward(self, u, v):
                return u[::2] + v[::2]

        foo = Foo()

        u, v = torch.randn(10), torch.randn(10)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimw = dimx * 2  # works
        ep = export(
            foo,
            (u, v),
            dynamic_shapes=({0: dimw}, {0: dimw}),
        )
        self.assertEqual(ep.module()(torch.randn(8), torch.randn(8)).size()[0], 4)

    def test_derived_dim_out_of_order(self):
        dimy = torch.export.Dim("dimy", min=5, max=7)
        dimx = dimy - 1  # out of order, effectively dimy = dimx + 1
        dimz = dimy + 1  # out of order, effectively dimz = dimx + 2

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1:] + z[2:]

        foo = Foo()

        u, v, w = torch.randn(5), torch.randn(6), torch.randn(7)
        ep = export(
            foo,
            (u, v, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 8, but got 5",
        ):
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(5))

        self.assertEqual(
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(8)).size()[0], 6
        )

    def test_derived_dim_out_of_order_repeat_derived(self):
        dimy = torch.export.Dim("dimy", min=5, max=7)
        dimx = dimy - 1  # out of order, effectively dimy = dimx + 1
        dimz = dimy + 1  # out of order, effectively dimz = dimx + 2
        dimx1 = dimx
        dimx2 = dimz - 2  # works, effectively = dimx

        class Foo(torch.nn.Module):
            def forward(self, x, y, z, x1, x2):
                return x + y[1:] + z[2:] + x1 + x2

        foo = Foo()

        u, v, w, u1, u2 = (
            torch.randn(5),
            torch.randn(6),
            torch.randn(7),
            torch.randn(5),
            torch.randn(5),
        )
        ep = export(
            foo,
            (u, v, w, u1, u2),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 6, but got 5",
        ):
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )

        ep = export(
            foo,
            (u, v, w, u, u),  # reused inputs
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 6, but got 5",
        ):
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )

    def test_specialize_derived_dim_roots(self):
        # dim & derived dim both specialize
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x.reshape([-1]) + y

        dy = Dim("dy", min=6)
        x, y = torch.randn(6, 2), torch.randn(12)
        dynamic_shapes = {
            "x": (dy - 6, 2),
            "y": (dy,),
        }
        try:
            export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)
            raise Exception(
                "export() call should have failed with dynamic shapes error."
            )
        except torch._dynamo.exc.UserError as exc:
            expected_error_msg = (
                "Specializations unexpectedly required \(dy\)!(.*\n)*.*"
                ".*dy - 6.*must be specialized to 6 because the guards generated for it are too complex(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                ".*dy = 12(.*\n)*.*"
            )
            self.assertTrue(re.search(expected_error_msg, exc.args[0]) is not None)
            self.assertTrue(
                "dy - 6 = 6" not in exc.args[0]
            )  # don't suggest fix for non-root dim

    def test_keep_composite_ops_invalid(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return torch.ops.aten.chunk.default(x, 3, 0)

        with self.assertRaisesRegex(
            RuntimeError, "aten.chunk.default is a mutating/aliasing op"
        ):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({}, _preserve_ops=(torch.ops.aten.chunk.default,))

        with self.assertRaisesRegex(
            RuntimeError,
            "aten.add.Tensor is not CompositeImplicitAutograd op, so we will preserve it as",
        ):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({}, _preserve_ops=(torch.ops.aten.add.Tensor,))

        with self.assertRaisesRegex(
            RuntimeError, "aten.sym_size.default is a metadata query function"
        ):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({}, _preserve_ops=(torch.ops.aten.sym_size.default,))

        with self.assertRaisesRegex(
            RuntimeError,
            "We can't detect aten.native_batch_norm.default as a functional op statically",
        ):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions(
                {}, _preserve_ops=(torch.ops.aten.native_batch_norm.default,)
            )

    def test_keep_composite_ops_linear_convd(self):
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x, y):
                x_conv = self.conv(x)
                y_conv_1d = self.conv1d(y)
                x_linear = self.linear(x_conv)
                return x_linear.cos() + y_conv_1d.sum()

        ep = torch.export.export(
            Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50))
        )
        ep_has_linear_convd = ep.run_decompositions(
            decomp_table={},
            _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY,
        )
        self.assertExpectedInline(
            str(ep_has_linear_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    linear = torch.ops.aten.linear.default(conv2d, c_linear_weight, c_linear_bias);  conv2d = c_linear_weight = c_linear_bias = None
    cos = torch.ops.aten.cos.default(linear);  linear = None
    sum_1 = torch.ops.aten.sum.default(conv1d);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        ep_has_convd = ep.run_decompositions(
            decomp_table=None,
            _preserve_ops=[
                torch.ops.aten.conv2d.default,
                torch.ops.aten.conv1d.default,
            ],
        )
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    addmm = torch.ops.aten.addmm.default(c_linear_bias, view, permute);  c_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(conv1d, []);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        ep_has_convd = ep_has_convd.run_decompositions(
            decomp_table=None, _preserve_ops=[torch.ops.aten.conv2d.default]
        )
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    convolution = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    addmm = torch.ops.aten.addmm.default(c_linear_bias, view, permute);  c_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution, []);  convolution = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

    def test_keep_composite_ops_linear_convd_for_training_ir(self):
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(20, 98))
                self.register_buffer("bias", torch.randn(20))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x, y):
                x_conv = self.conv(x)
                y_conv_1d = self.conv1d(y)
                x_linear = self.linear(x_conv)
                return x_linear.cos() + y_conv_1d.sum()

        ep = torch.export._trace._export_for_training(
            Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50))
        )
        ep_has_linear_convd = ep.run_decompositions(
            decomp_table={},
            _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY,
        )
        self.assertExpectedInline(
            str(ep_has_linear_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    convolution = torch.ops.aten.convolution.default(x, p_conv_weight, p_conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  x = p_conv_weight = p_conv_bias = None
    convolution_1 = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(convolution, [31680, 98]);  convolution = None
    t = torch.ops.aten.t.default(b_linear_weight);  b_linear_weight = None
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, t);  b_linear_bias = view = t = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.default(convolution_1);  convolution_1 = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        ep_has_convd = ep.run_decompositions(
            decomp_table=None,
            _preserve_ops=[
                torch.ops.aten.conv2d.default,
                torch.ops.aten.conv1d.default,
            ],
        )
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    convolution = torch.ops.aten.convolution.default(x, p_conv_weight, p_conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  x = p_conv_weight = p_conv_bias = None
    convolution_1 = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(convolution, [31680, 98]);  convolution = None
    t = torch.ops.aten.t.default(b_linear_weight);  b_linear_weight = None
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, t);  b_linear_bias = view = t = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.default(convolution_1);  convolution_1 = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        ep_has_convd = ep_has_convd.run_decompositions(
            decomp_table=None, _preserve_ops=[torch.ops.aten.conv2d.default]
        )
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    convolution = torch.ops.aten.convolution.default(x, p_conv_weight, p_conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  x = p_conv_weight = p_conv_bias = None
    convolution_1 = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(convolution, [31680, 98]);  convolution = None
    permute = torch.ops.aten.permute.default(b_linear_weight, [1, 0]);  b_linear_weight = None
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, permute);  b_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution_1, []);  convolution_1 = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

    def test_derived_dim_out_of_order_simplified(self):
        _dimz = torch.export.Dim("_dimz", min=6, max=8)
        dimy = _dimz - 1
        dimx = dimy - 1
        dimz = torch.export.Dim("dimz", min=6, max=8)  # doesn't work, should be = _dimz

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1:] + z[2:]

        foo = Foo()
        u, v, w = torch.randn(5), torch.randn(6), torch.randn(7)
        try:
            export(
                foo,
                (u, v, w),
                dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
            )
        except torch._dynamo.exc.UserError as exc:
            expected_error_msg = (
                "Constraints violated \(dimz\)!(.*\n)*.*"
                "The values of dimz.*must always be related to the values of _dimz - 2.*by.*(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dimz = _dimz"
            )
            self.assertTrue(re.search(expected_error_msg, exc.args[0]) is not None)
            # don't suggest fix for non-root dims, and no need to update root here
            self.assertTrue("_dimz - 2 = Dim(" not in exc.args[0])
            self.assertTrue("_dimz - 1 = _dimz - 1" not in exc.args[0])
            self.assertTrue("_dimz = Dim(" not in exc.args[0])

        dimz = dimx + 2  # works, effectively = _dimz
        ep = export(
            foo,
            (u, v, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 8, but got 5",
        ):
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(5))

        self.assertEqual(
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(8)).size()[0], 6
        )

    def test_simple_export_for_training(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        eager_model = Foo()
        ep_for_training = torch.export._trace._export_for_training(
            eager_model, (torch.ones(2, 2),)
        )
        self.assertExpectedInline(
            str(ep_for_training.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    return (linear,)""",
        )
        gm = ep_for_training.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    linear_weight = self.linear.weight
    linear_bias = self.linear.bias
    linear = torch.ops.aten.linear.default(x, linear_weight, linear_bias);  x = linear_weight = linear_bias = None
    return pytree.tree_unflatten((linear,), self._out_spec)""",
        )

        self.assertTrue(
            torch.allclose(gm(torch.ones(2, 2)), eager_model(torch.ones(2, 2)))
        )

    def test_export_for_training_with_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(4, 4))

            def forward(self, x):
                x.add_(5)
                self.buffer.add_(5)
                return x + self.buffer

        eager_model_for_export = Foo()
        eager_model_for_testing = Foo()
        ep_for_training = torch.export._trace._export_for_training(
            eager_model_for_export, (torch.ones(4, 4),)
        )
        self.assertExpectedInline(
            str(ep_for_training.graph_module.code).strip(),
            """\
def forward(self, b_buffer, x):
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    add__1 = torch.ops.aten.add_.Tensor(b_buffer, 5);  b_buffer = None
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    return (add,)""",
        )
        gm = ep_for_training.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    buffer = self.buffer
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    add__1 = torch.ops.aten.add_.Tensor(buffer, 5);  buffer = None
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

        self.assertTrue(
            torch.allclose(
                gm(torch.ones(4, 4)), eager_model_for_testing(torch.ones(4, 4))
            )
        )

    def test_export_for_training_with_dynamic_shapes(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(4, 4))

            def forward(self, x):
                x.add_(5)
                self.buffer.add_(5)
                return x + self.buffer.sum()

        eager_model_for_export_training = Foo()
        eager_model_for_export_inference = Foo()
        eager_model_for_testing = Foo()
        ep_for_training = torch.export._trace._export_for_training(
            eager_model_for_export_training,
            (torch.ones(4, 4),),
            dynamic_shapes=({0: Dim("x")},),
        )

        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(torch.ones(2, 4)),
                eager_model_for_testing(torch.ones(2, 4)),
            )
        )

        ep_for_real = export(
            eager_model_for_export_inference,
            (torch.ones(4, 4),),
            dynamic_shapes=({0: Dim("x")},),
        )

        self.assertEqual(
            str(ep_for_training.range_constraints), str(ep_for_real.range_constraints)
        )

    def test_export_for_training_with_container_type(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(4, 4))

            def forward(self, container):
                x = container[0][0]
                y = container[0][1]
                x.add_(5)
                y.add_(5)
                return x + y + self.buffer.sum()

        eager_model = Foo()
        ep_for_training = torch.export._trace._export_for_training(
            eager_model,
            ([torch.ones(4, 4), torch.ones(4, 4)],),
        )

        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(
                    ([torch.ones(4, 4), torch.ones(4, 4)]),
                ),
                eager_model(([torch.ones(4, 4), torch.ones(4, 4)])),
            )
        )

    def test_export_for_training_run_decomp(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(2, 2))
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                self.buffer.add_(5)
                return self.linear(x) + self.buffer.sum()

        eager_model = Foo()
        ep_for_training = torch.export._trace._export_for_training(
            eager_model,
            (torch.ones(2, 2),),
        )
        ep_for_inference = ep_for_training.run_decompositions()
        self.assertExpectedInline(
            str(ep_for_inference.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, b_buffer, x):
    add = torch.ops.aten.add.Tensor(b_buffer, 5);  b_buffer = None
    t = torch.ops.aten.t.default(p_linear_weight);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, x, t);  p_linear_bias = x = t = None
    sum_1 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(addmm, sum_1);  addmm = sum_1 = None
    return (add, add_1)""",
        )

    def test_derived_dim_out_of_order_simplified_repeat_non_derived(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, y1, z):
                return x + y[1:] + y1[1:] + z[2:]

        foo = Foo()

        u, v, v1, w = torch.randn(5), torch.randn(6), torch.randn(6), torch.randn(7)
        _dimz = torch.export.Dim("_dimz", min=6, max=8)
        dimy = _dimz - 1
        dimx = dimy - 1
        dimz = dimx + 2  # works, effectively = _dimz
        ep = export(
            foo,
            (u, v, v1, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 7, but got 5",
        ):
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(5),
                torch.randn(8),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(7),
                torch.randn(8),
            ).size()[0],
            6,
        )

    def test_static_dim_constraints(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(6, 4)

            def forward(self, x, y, z):
                x0 = self.l(x) + y[1:]
                return x0, z * 2.0

        foo = Foo()
        inputs = (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 3))
        dx = Dim("dx", min=3, max=6)
        dy = dx + 1
        dz = Dim("dz", min=3, max=6)

        # all of these should be fine
        for dynamic_shapes in [
            ({0: dx, 1: 6}, {0: dy, 1: 4}, {0: dz, 1: 3}),
            ((dx, None), (dy, 4), (dz, 3)),
            ((None, 6), (5, None), (None, None)),
            ((4, 6), {0: None, 1: 4}, {0: None, 1: 3}),
        ]:
            ep = export(foo, inputs, dynamic_shapes=dynamic_shapes)
            self.assertEqual(foo(*inputs), ep.module()(*inputs))

        # check range_constraints - static dims shouldn't be present
        ep = export(foo, inputs, dynamic_shapes=((dx, None), (dy, 4), (dz, 3)))
        self.assertEqual(len(ep.range_constraints), 3)
        for vr in ep.range_constraints.values():
            self.assertTrue(vr.lower < vr.upper)

        # check raised errors
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 5 does not match input size of 4, for .*",
        ):
            _ = export(foo, inputs, dynamic_shapes=((5, None), None, None))
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 9 does not match input size of 6, for .*",
        ):
            _ = export(foo, inputs, dynamic_shapes=((dx, 9), (dy, 4), (3, 3)))

    def test_dim_1_2(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * 2

        dx = Dim("dx", min=1, max=2)
        ep = export(Foo(), (torch.randn(2, 2),), dynamic_shapes=({0: dx, 1: None},))
        ep.module()(torch.randn(1, 2))
        ep.module()(torch.randn(2, 2))
        with self.assertRaisesRegex(
            RuntimeError, "Expected input at .* to be <= 2, but got 3"
        ):
            ep.module()(torch.randn(3, 2))
        vr = list(ep.range_constraints.values())[0]
        self.assertEqual(vr.lower, 1)
        self.assertEqual(vr.upper, 2)

    def test_derived_dim_1_2(self):
        class Bar(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1:]

        dx = Dim("dx", min=1, max=2)
        ep = export(
            Bar(),
            (torch.randn(2, 2), torch.randn(3, 2)),
            dynamic_shapes=({0: dx, 1: None}, {0: dx + 1, 1: None}),
        )
        ep.module()(torch.randn(1, 2), torch.randn(2, 2))
        range_lower_bounds = sorted(vr.lower for vr in ep.range_constraints.values())
        range_upper_bounds = sorted(vr.upper for vr in ep.range_constraints.values())
        self.assertEqual(range_lower_bounds, [1, 2])
        self.assertEqual(range_upper_bounds, [2, 3])

    def test_dynamic_shapes_builder_basic(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[0] + z["k"]

        m = M()

        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (x, y, z)

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, dynamic_shapes=shapes_collection)
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_dynamic_shapes_builder_kwargs(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[0] + z["k"]

        m = M()

        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (x,)
        kwargs = {"z": z, "y": y}

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, kwargs=kwargs, dynamic_shapes=shapes_collection)
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    # retracing doesn't seem to like dataclass registration,
    # raising a dynamo error in fx_pytree.tree_flatten_spec
    @testing.expectedFailureRetraceability
    def test_dynamic_shapes_builder_pytree(self):
        torch.export.register_dataclass(
            Inp,
            serialized_type_name="test_dynamic_shapes_builder_pytree.Inp",
        )

        class M(torch.nn.Module):
            def forward(self, inp: Inp):
                return inp.x + inp.y[0] + inp.z["k"]

        m = M()
        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (Inp(x, y, z),)

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, dynamic_shapes=shapes_collection.dynamic_shapes(m, args))
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    def test_torch_check_eq_commutativity(self):
        class M1(torch.nn.Module):
            def forward(self, x1, x2, x3, y):
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # instead of: torch._check((z2 + z3) == z1)
                torch._check(z1 == (z2 + z3))
                if z2 + z3 == z1:
                    return y * 2
                else:
                    return y + 3

        export(
            M1(),
            (torch.tensor(6), torch.tensor(3), torch.tensor(3), torch.randn(1)),
        )

        class M2(torch.nn.Module):
            def forward(self, x1, x2, x3, y):
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # instead of: torch._check((z2 + z3) != z1)
                torch._check(z1 != (z2 + z3))
                if z2 + z3 == z1:
                    return y * 2
                else:
                    return y + 3

        export(
            M2(),
            (torch.tensor(6), torch.tensor(6), torch.tensor(6), torch.randn(1)),
        )

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.shape[0]
                if z > 2:
                    return x.cos()
                else:
                    return x.sin()

        with self.assertRaisesRegex(
            (
                torchdynamo.exc.UserError,
                torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
            ),
            "Could not guard on data-dependent expression",
        ):
            _ = export(M(), (torch.tensor([2, 3, 5]),))

    def test_if_functional(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                z = x + 4
                z.add_(4)
                y = z.view(x.shape)
                return x.cos() + y.cos()

        foo = Module()
        gm = export(foo, (torch.tensor([2, 3, 5]),))

        view_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # No more inplace mutation
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph.",
                )
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.view.default
            ):
                view_count += 1

        # There should be nonzero view nodes in the graph
        self.assertTrue(view_count > 0)

    def test_export_mod_constraints(self):
        class BasicDynamiShapeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.shape[0] - 1, -1)

        m = BasicDynamiShapeModel()
        a = torch.randn(3, 4)
        dim0_x = torch.export.Dim("dim0_x", min=3)
        dim1_x = torch.export.Dim("dim1_x", max=8000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Specializations unexpectedly required"
                ".*\n.*\\[0\\] must be specialized to 3.*guards.*too complex(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dim0_x = 3(.*\n)*.*"
                "dim1_x = 2\\*_dim1_x"
            ),
        ):
            torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)
        dim0_x = None
        dim1_x = 2 * torch.export.Dim("_dim1_x", max=4000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        em = torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)
        x = torch.randn(3, 5)
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected.*shape\\[1\\] = 5 to be of the form 2\\*s1, where s1 is an integer",
        ):
            em.module()(x)

    def test_not_correct_dim(self):
        def f(x):
            return x.cos()

        def g(x):
            return x + 4

        inp_for_f = torch.tensor(5)
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError, "Cannot mark 0-dimension tensors to be dynamic"
        ):
            constraints = [dynamic_dim(inp_for_f, 0)]

        inp_for_f_mul_dim = torch.ones(5, 5)
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "Expected the dimension passed to dynamic_dim to be in the range \\[0:1\\]",
        ):
            constraints = [dynamic_dim(inp_for_f_mul_dim, 2)]

        inp_for_g = 4
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError, "Expected tensor as input to dynamic_dim"
        ):
            constraints = [dynamic_dim(inp_for_g, 0)]

    @testing.expectedFailureRetraceability  # T183144629
    def test_map(self):
        class Module(torch.nn.Module):
            def forward(self, xs, y, z):
                def body(x, y, z):
                    return x + y + z

                return map(body, xs, y, z)

        list_tensor_map = Module()
        inps = (torch.ones(6, 4), torch.tensor(5), torch.tensor(4))
        self._test_export_same_as_eager(list_tensor_map, inps)

    @unittest.expectedFailure
    def test_crop_like(self):
        # https://fb.workplace.com/groups/1405155842844877/posts/8195050017188725/

        # Minimal crop code copied from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional
        class CropLike(torch.nn.Module):
            def forward(self, image, crop_height, crop_width):
                c, image_height, image_width = image.shape
                crop_top = int(round((image_height - crop_height) / 2.0))
                crop_left = int(round((image_width - crop_width) / 2.0))
                return image[
                    ...,
                    crop_top : crop_top + crop_height,
                    crop_left : crop_left + crop_width,
                ]

        crop = CropLike()
        imagew = Dim("width")
        imageh = Dim("height")
        dynamic_dims = {
            "image": {0: None, 1: imageh, 2: imagew},
            "crop_height": None,
            "crop_width": None,
        }
        args = (torch.rand(3, 512, 512), 150, 150)
        ecrop = export(crop, args=args, dynamic_shapes=dynamic_dims)

        args = (torch.rand(3, 700, 700), 150, 150)
        self.assertEqual(ecrop.module()(*args), ecrop(*args))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_export_func_with_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, kw1, kw2):
                return arg1 + arg2, kw1 + kw2

        kw_func = Module()
        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs = {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_export_func_with_pytree_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b):
                return arg1 + a["kw1"] + b[0], arg2 + a["kw2"] + b[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {
            "a": {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)},
            "b": [torch.ones(2, 3), torch.ones(3, 4)],
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_export_func_with_default_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b=1):
                return arg1 + arg2, a["kw1"] + a["kw2"] + b

        kw_func = Module()

        class Module2(torch.nn.Module):
            def forward(self, arg1, arg2, a=1, b=2):
                return arg1 + a, arg2 + b

        kw_func2 = Module2()

        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs1 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}}
        kwargs2 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}, "b": 2}
        self._test_export_same_as_eager(kw_func, args, kwargs1)
        self._test_export_same_as_eager(kw_func, args, kwargs2)
        kwargs3 = {"b": 1}
        self._test_export_same_as_eager(kw_func2, args, kwargs3)

    def test_export_func_with_var_postional_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args):
                return arg1 + args[0], arg2 + args[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        self._test_export_same_as_eager(kw_func, args)

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_export_func_with_keyword_only_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2):
                return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_export_func_with_var_keyword_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                return (
                    arg1 + args[0] + kw1 + kwargs["kw3"],
                    arg2 + args[1] + kw2 + kwargs["kw4"],
                )

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {
            "kw1": torch.ones(2, 3),
            "kw2": torch.ones(3, 4),
            "kw3": torch.ones(2, 3),
            "kw4": torch.ones(3, 4),
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_unbacked_slice(self):
        class M(torch.nn.Module):
            def forward(self, scores, score_thr, topk: torch.Tensor, results=None):
                valid_mask = scores > score_thr
                scores = scores[valid_mask]
                valid_idxs = torch.nonzero(valid_mask).to(scores.device)

                num_topk = torch.minimum(topk, torch.tensor(valid_idxs.shape[0])).item()
                torch._check_is_size(num_topk)
                torch._check(scores.shape[0] >= num_topk)
                scores, idxs = scores.sort(descending=True)
                scores = scores[:num_topk]
                topk_idxs = valid_idxs[idxs[:num_topk]]
                keep_idxs, labels = topk_idxs.unbind(dim=1)

                return scores, labels, keep_idxs

        score = torch.tensor(
            [[0.1, 0.3, 0.2], [0.12, 0.7, 0.9], [0.02, 0.8, 0.08], [0.4, 0.1, 0.08]]
        )
        bbox_pred = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.1, 0.1], [0.5, 0.1]])
        score_thr = 0.15
        nms_pre = torch.tensor(4)
        inputs = (score, score_thr, nms_pre, dict(bbox_pred=bbox_pred))

        ep = torch.export.export(M(), inputs)
        orig_res = M()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    def test_unflatten_asserts(self):
        # TODO: strict-export fails
        class M1(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()

                torch._check_is_size(b)
                torch._check(b < y.size(0))
                return y[:b]

        class M3(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()

                torch._check_is_size(b)
                torch._check(b < y.size(0) * 2)
                return y[:b]

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = M1()
                self.m3 = M3()

            def forward(self, x, y):
                return self.m1(x, y) + self.m3(x, y)

        inputs = (torch.tensor(3), torch.randn(10))

        ep = torch.export.export(
            M2(), inputs, dynamic_shapes={"x": None, "y": (Dim("moo"),)}, strict=False
        )
        orig_res = M2()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

        unflattened = torch.export.unflatten(ep)
        ep_res = unflattened(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193693183
    def test_export_func_with_var_keyword_pytree_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                return (
                    arg1 + arg2[0][0] + args[0] + kw1[0] + kwargs["kw3"][0],
                    arg2[1] + args[1] + kw2 + kwargs["kw4"],
                )

        kw_func = Module()
        args = (
            torch.ones(2, 3),
            [(torch.ones(2, 3),), torch.ones(3, 4)],
            torch.ones(2, 3),
            torch.ones(3, 4),
        )
        kwargs = {
            "kw1": (torch.ones(2, 3),),
            "kw2": torch.ones(3, 4),
            "kw3": (torch.ones(2, 3), torch.ones(3, 4)),
            "kw4": torch.ones(3, 4),
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureSerDer  # we don't save placeholder metadata
    @testing.expectedFailureNonStrict
    @testing.expectedFailureTrainingIRToRunDecomp  # T193692674
    def test_linear_conv(self):
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                x_conv = self.conv(x)
                x_linear = self.linear(x_conv)
                return x_linear.cos()

        ep = export(Foo(), (torch.randn(20, 16, 50, 100),))
        for node in ep.graph.nodes:
            if (
                node.op == "placeholder"
                and node.name in ep.graph_signature.inputs_to_buffers
                or node.name in ep.graph_signature.inputs_to_parameters
            ):
                self.assertTrue("source_fn_stack" in node.meta)

    def test_export_api_with_dynamic_shapes(self):
        from torch.export import Dim, dims, export

        # pass dynamic shapes of inputs [args]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={k: {0: batch} for k in ["x", "y"]},
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        efoo = export(
            foo, inputs, kwinputs, dynamic_shapes={k: {0: batch} for k in ["x", "y"]}
        )
        self.assertEqual(
            efoo.module()(*inputs, **kwinputs).shape, foo(*inputs, **kwinputs).shape
        )

        # pass dynamic shapes of inputs [partial, error]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(batch\\)!(.*\n)*.*"
                "batch was inferred to be a constant(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = 10"
            ),
        ):
            export(
                foo,
                inputs,
                kwinputs,
                dynamic_shapes={"x": {0: batch}, "y": None},
            )

        # pass dynamic shapes of inputs [module]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [bounds, mostly shared]
        foo = Foo()
        inputs = (torch.randn(10, 3, 3), torch.randn(10, 3, 3))
        batch = Dim("batch", min=8, max=64)
        size = Dim("size")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={
                "x": (batch, size, size),
                "y": (batch, size, size),
            },
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, s1, s1])", "torch.Size([s0, s1, s1])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [multiple, mostly distinct]
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        efoo = export(
            Foo(),
            inputs,
            dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, s1, s2])", "torch.Size([s0, s2, s5])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [dict]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs["x"], inputs["y"])

        foo = Foo()
        inputs = ({"x": torch.randn(10, 2, 3), "y": torch.randn(10, 3, 4)},)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": {k: {0: batch} for k in ["x", "y"]}}
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [list]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs[0], inputs[1])

        foo = Foo()
        inputs = ([torch.randn(10, 2, 3), torch.randn(10, 3, 4)],)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": [{0: batch} for _ in range(2)]}
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [dataclass]

        # TODO(avik): This part of the test should have failed both serde and retracing
        # but these failures are hidden because of the local import of `export` in this test.
        # The serde failure is benign, and easily avoided by moving the dataclass definition
        # to the top-level. OTOH the retracing failure needs further investigation.
        @dataclass
        class DataClass:
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            DataClass,
            serialized_type_name="test_export_api_with_dynamic_shapes.DataClass",
        )

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs.a, inputs.b)

        foo = Foo()
        inputs = (DataClass(a=torch.randn(10, 2, 3), b=torch.randn(10, 3, 4)),)
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"inputs": [{0: batch}, {0: batch}]},
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )

        # pass dynamic shapes of inputs [pytree-registered classes]
        if HAS_TORCHREC:
            # skipping tests if torchrec not available
            class Foo(torch.nn.Module):
                def forward(self, kjt) -> torch.Tensor:
                    return kjt.values() + 0, kjt.offsets() + 0

            foo = Foo()
            kjt = KeyedJaggedTensor(
                values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                keys=["index_0", "index_1"],
                lengths=torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3]),
                offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
            )
            inputs = (kjt,)
            dim = Dim("dim")
            dim_plus_one = Dim("dim_plus_one")
            efoo = torch.export.export(
                foo,
                inputs,
                dynamic_shapes={"kjt": [{0: dim}, None, {0: dim}, {0: dim_plus_one}]},
            )
            self.assertEqual(
                [out.shape for out in efoo.module()(*inputs)],
                [out.shape for out in foo(*inputs)],
            )

        # pass dynamic shapes of inputs [distinct, error]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, K2, N = dims("batch", "M", "K1", "K2", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(K2\\)!(.*\n)*.*"
                "K2.*and.*K1.*must always be equal(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "K2 = K1"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, K2, N)},
            )

        # pass dynamic shapes of inputs [specialized, error]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, N = dims("batch", "M", "K1", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(K1\\)!(.*\n)*.*"
                "K1 was inferred to be a constant(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "K1 = 3"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, None, N)},
            )

        # pass dynamic shapes of inputs [guards, error]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] < 16 and y.shape[1] % 3 == 0:
                    return torch.matmul(x, y)
                else:
                    return x + y

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated.*!(.*\n)*.*"
                "Not all values of K.*satisfy the generated guard(.*\n)*.*"
                "Not all values of batch.*satisfy the generated guard(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = Dim\\('batch', max=15\\)(.*\n)*.*"
                "K = 3\\*_K"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
            )

    def test_suggested_fixes_new_roots(self):
        from torch.export import dims

        # suggested fixes should introduce new root dim for modulo guard
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                # dy = 3 * _dx
                # dx = 3 * _dx - 1
                # dz = 3 * _dx + 2
                # suggested fixes results will look something like
                # {"dx": {"eq": 3*_dx-1, "min": 5, "max": 36}, "dy": {"eq": dx+1}, ...}
                if x.shape[0] >= 5 and x.shape[0] <= 36 and y.shape[0] % 3 == 0:
                    return x + y[1:] + z[3:]

        foo = Foo()
        inputs = (
            torch.randn(
                11,
            ),
            torch.randn(
                12,
            ),
            torch.randn(
                14,
            ),
        )
        dx, dy, dz = dims("dx", "dy", "dz")
        dynamic_shapes = {
            "x": (dx,),
            "y": (dy,),
            "z": (dz,),
        }
        with self.assertRaisesRegex(  # figure out regex later
            torch._dynamo.exc.UserError,
            (
                "Constraints violated.*!(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                "_dx = Dim\(\\'_dx\\', max=12\)(.*\n)*.*"
                "dx = 3\*_dx - 1(.*\n)*.*"
                "dy = 3\*_dx(.*\n)*.*"
                "dz = 3\*_dx \+ 2"
            ),
        ):
            export(Foo(), inputs, dynamic_shapes=dynamic_shapes)
        # retry export
        _dx = Dim("_dx", min=2, max=12)
        dynamic_shapes = {"x": (3 * _dx - 1,), "y": (3 * _dx,), "z": (3 * _dx + 2,)}
        export(Foo(), inputs, dynamic_shapes=dynamic_shapes)

    def test_refine_dynamic_shapes_from_suggested_fixes(self):
        from torch.export.dynamic_shapes import (
            refine_dynamic_shapes_from_suggested_fixes,
        )

        def helper(model, inputs, dynamic_shapes):
            # export, fail, parse & refine suggested fixes, re-export
            try:
                export(Foo(), inps, dynamic_shapes=dynamic_shapes)
                raise Exception("should have raised constraint violation error")
            except torch._dynamo.exc.UserError as exc:
                new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                    exc.msg, dynamic_shapes
                )
                export(Foo(), inps, dynamic_shapes=new_shapes)
                return new_shapes

        # specialize dims + derived dims
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                x0 = x + y[1:] + z[2:]
                x1 = x @ torch.randn(4, 4)
                return x0, x1

        inps = (
            torch.randn(
                4,
            ),
            torch.randn(
                5,
            ),
            torch.randn(
                6,
            ),
        )
        dx = Dim("dx", max=16)
        dynamic_shapes = {"x": (dx,), "y": (dx + 1,), "z": (dx + 2,)}
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0], 4)
        self.assertEqual(new_shapes["z"][0], 6)

        # refine lower, upper bound
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] >= 6 and y.shape[0] <= 16:
                    return x * 2.0, y + 1

        inps = (torch.randn(16), torch.randn(12))
        dynamic_shapes = {"x": (Dim("dx"),), "y": (Dim("dy"),)}
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0].min, 6)
        self.assertEqual(new_shapes["y"][0].max, 16)

        # divisiblity, will introduce new root
        class Foo(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] >= 9:
                    return x.reshape([-1, 3])

        inps = (
            torch.randn(
                15,
            ),
        )
        dynamic_shapes = ((Dim("dx"),),)
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        dim = new_shapes[0][0]
        root = dim.root
        self.assertEqual(dim.fn(2), 6)
        self.assertEqual(root.min, 3)

        # turn dim into derived dim/relation
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[4:]

        inps = (torch.randn(6, 4), torch.randn(10, 4))
        dynamic_shapes = {
            "x": (Dim("dx0"), Dim("dx1")),
            "y": (Dim("dy0"), Dim("dy1")),
        }
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0], new_shapes["y"][0].root)  # dy0 = dx0 + 4
        self.assertEqual(new_shapes["y"][0].fn(5), 9)
        self.assertEqual(new_shapes["x"][1], new_shapes["y"][1])  # dx1 = dy1

        # nested dynamic shapes spec
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x0 = x[0]["data"] + x[1] + x[2][2:]
                x1 = y["a"] @ torch.randn(4, 4)
                x2 = y["b"] @ torch.randn(6, 6)
                return x0, x1, x2

        inps = (
            [
                {"data": torch.randn(4, 4)},
                torch.randn(4, 4),
                torch.randn(6, 4),
            ],
            {
                "a": torch.randn(8, 4),
                "b": torch.randn(9, 6),
            },
        )
        dynamic_shapes = {
            "x": [
                {"data": (Dim("dx00"), Dim("dx01"))},
                (Dim("dx10"), Dim("dx11")),
                (Dim("dx20"), Dim("dx21")),
            ],
            "y": {
                "a": (Dim("dya0"), Dim("dya1")),
                "b": (Dim("dyb0"), Dim("dyb1")),
            },
        }
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(
            new_shapes["x"][0]["data"][0], new_shapes["x"][1][0]
        )  # dx10 = dx00
        self.assertEqual(
            new_shapes["x"][2][0].root, new_shapes["x"][0]["data"][0]
        )  # dx20 = dx00 + 2
        self.assertEqual(new_shapes["x"][2][0].fn(10), 12)
        self.assertEqual(
            new_shapes["x"][0]["data"][1], new_shapes["x"][1][1]
        )  # dx11 = dx01
        self.assertEqual(new_shapes["y"]["a"][1], 4)
        self.assertEqual(new_shapes["y"]["b"][1], 6)
        self.assertEqual(new_shapes["y"]["b"][0].__name__, "dyb0")  # unchanged

    def test_dynamic_shapes_spec_with_pytree(self):
        from torch.export import Dim, export
        from torch.utils._pytree import tree_map

        inputs = {
            "tensor": torch.randn(3),
            "dict_of_tensors": {k: torch.randn(3) for k in ["A", "B", "C", "D"]},
            "list_of_tensors": [torch.randn(3) for _ in range(4)],
        }

        batch = Dim("batch")
        # uniformly specify dynamic shapes for all inputs
        spec = tree_map(lambda x: {0: batch}, inputs)

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return (
                    inputs["tensor"]
                    + inputs["dict_of_tensors"]["A"]
                    + inputs["list_of_tensors"][0]
                )

        ep = export(Foo(), (inputs,), dynamic_shapes={"inputs": spec})
        input_shapes = [
            str(node.meta["val"].shape)
            for node in ep.graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(len(input_shapes), 9)
        self.assertTrue(all(shape == "torch.Size([s0])" for shape in input_shapes))

    def test_error_does_not_reference_eager_fallback(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.shape[0]
                if z > 2:
                    return x.cos()
                else:
                    return x.sin()

        fn_ddo = Module()
        if is_non_strict_test(self._testMethodName):
            error = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
            error_msg = r"Could not guard on data-dependent expression"
        else:
            error = torchdynamo.exc.UserError
            error_msg = r"^(?!.*fall back to eager).*"
        with self.assertRaisesRegex(error, error_msg):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),))

    def test_pytree_register_data_class(self):
        @dataclass
        class MyDataClass:
            x: int
            y: int
            z: int = None

        dt = MyDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertTrue(spec, LeafSpec())
        self.assertTrue(len(flat) == 1)

        register_dataclass_as_pytree_node(
            MyDataClass,
            serialized_type_name="test_pytree_register_data_class.MyDataClass",
        )

        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(MyDataClass, [["x", "y"], ["z"]], [LeafSpec(), LeafSpec()]),
        )
        self.assertEqual(flat, [3, 4])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

        @dataclass
        class MyOtherDataClass:  # the pytree registration don't allow registering the same class twice
            x: int
            y: int
            z: int = None

        # Override the registration with keep none fields
        register_dataclass_as_pytree_node(
            MyOtherDataClass,
            return_none_fields=True,
            serialized_type_name="test_pytree_regster_data_class.MyOtherDataClass",
        )

        dt = MyOtherDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyOtherDataClass,
                [["x", "y", "z"], []],
                [LeafSpec(), LeafSpec(), LeafSpec()],
            ),
        )
        self.assertEqual(flat, [3, 4, None])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyOtherDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_register_nested_data_class(self):
        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            xy: Inner
            ab: Inner

        xy = Inner(1, 2)
        ab = Inner(3, 4)
        dt = Outer(xy, ab)
        inp = {"dt1": (dt, ({},)), "dt2": ((torch.ones(1),), dt)}

        register_dataclass_as_pytree_node(
            Inner, serialized_type_name="test_pytree_register_nested_data_class.Inner"
        )
        register_dataclass_as_pytree_node(
            Outer, serialized_type_name="test_pytree_register_nested_data_class.Outer"
        )

        flat, spec = tree_flatten(inp)
        self.assertEqual(flat, [1, 2, 3, 4, torch.ones(1), 1, 2, 3, 4])

        unflat = tree_unflatten(flat, spec)
        self.assertEqual(unflat, inp)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_param_util(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        ep = export(Basic(), (torch.randn(5, 10),))
        num_params = 0
        params = []
        for node in ep.graph.nodes:
            if is_param(ep, node):
                num_params += 1
                params.append(get_param(ep, node))
        self.assertEqual(num_params, 2)
        self.assertEqual(params[0].shape, [1, 10])  # weight
        self.assertEqual(params[1].shape, [1])  # bias

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700631
    def test_buffer_util(self):
        ep = export(
            torch.nn.BatchNorm2d(100, affine=False), (torch.ones(20, 100, 35, 45),)
        )
        num_buffer = 0
        buffer = []

        for node in ep.graph.nodes:
            if is_buffer(ep, node):
                num_buffer += 1
                buffer.append(get_buffer(ep, node))
        self.assertEqual(num_buffer, 3)

        self.assertEqual(buffer[0].shape, torch.Size([100]))  # running_mean
        self.assertEqual(buffer[1].shape, torch.Size([100]))  # running_var
        self.assertEqual(buffer[2].shape, torch.Size([]))  # num_batches_tracked

    @testing.expectedFailureTrainingIRToRunDecomp  # T193701564
    def test_export_dynamo_config(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.lstm(inputs)

        config = DEFAULT_EXPORT_DYNAMO_CONFIG
        mod = MyModule()

        @contextmanager
        def _patch_config(kwargs):
            orig_config_dict = dataclasses.asdict(config)

            try:
                for k, v in kwargs.items():
                    setattr(config, k, v)
                yield
            finally:
                for k, v in orig_config_dict.items():
                    setattr(config, k, v)

        inp = (torch.rand(5, 4),)
        exported_program = export(mod, inp, strict=True)

        with _patch_config({"allow_rnn": False}):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "TorchDynamo purposely graph breaks on RNN, GRU, LSTMs",
            ):
                _ = export(mod, inp, strict=True)

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_device_to_static(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(Module(), (torch.tensor(1, device="cpu"),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_device_to_dynamic(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(
            Module(),
            (torch.tensor([1, 2], device="cpu"),),
            dynamic_shapes={"x": {0: Dim("i")}},
        )
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_device_to_mutation(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.to("cpu")
                y.add_(1)
                return y, x

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            export(Module(), (torch.tensor(1, device="cpu"),))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_float_conversion(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.float()

        ep = export(Module(), (torch.tensor(1, dtype=torch.float),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193700396
    def test_device_to_mutation_float(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.float()
                y.add_(1)
                return y, x

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            export(Module(), (torch.tensor(1, dtype=torch.float),))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193692674
    def test_module(self):
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a, b = x
                a_conv = self.conv(a)
                a_linear = self.linear(a_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return (
                    a_linear.cos() + b_linear.sin(),
                    a_linear.sin() + b_linear.cos(),
                )

        inp_container = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[0], ep_rexported.module()(*inp_test)[0]
            )
        )
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[1], ep_rexported.module()(*inp_test)[1]
            )
        )

    @testing.expectedFailureTrainingIRToRunDecomp  # T193701564
    def test_module_with_dict_container_inp_out(self):
        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a1, a2 = x["a"]
                b = x["b"]
                a1_conv = self.conv(a1)
                a1_linear = self.linear(a1_conv)
                a2_conv = self.conv(a2)
                a2_linear = self.linear(a2_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return {
                    "a": a1_linear.cos() + b_linear.sin(),
                    "b": a2_linear.sin() + b_linear.cos(),
                }

        inp_container = (
            {
                "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),
                "b": torch.randn(20, 16, 50, 100),
            },
        )

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = (
            {
                "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),
                "b": torch.randn(20, 16, 50, 100),
            },
        )

        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)["a"], ep_rexported.module()(*inp_test)["a"]
            )
        )
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)["b"], ep_rexported.module()(*inp_test)["b"]
            )
        )

    def test_args_type_checked(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.rand(2, 2)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "to be a tuple"):
            # Intentionally not wrapping `inp` in a tuple to trigger the error
            _ = export(M(), inp)

    def test_decomp_batch_norm_functional_predispatch(self):
        class ConvBatchnorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return (x,)

        mod = ConvBatchnorm()
        mod.eval()
        inp = torch.randn(1, 1, 3, 3)

        gm = torch.export._trace._export(mod, (inp,), pre_dispatch=True).module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    bn_weight = self.bn.weight
    bn_bias = self.bn.bias
    bn_running_mean = self.bn.running_mean
    bn_running_var = self.bn.running_var
    conv2d = torch.ops.aten.conv2d.default(x, conv_weight, conv_bias);  x = conv_weight = conv_bias = None
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, 0.1, 1e-05);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    getitem = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
    return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )

        mod.train()
        gm_train = _export(mod, (inp,), pre_dispatch=True).module()
        self.assertExpectedInline(
            str(gm_train.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    bn_weight = self.bn.weight
    bn_bias = self.bn.bias
    bn_running_mean = self.bn.running_mean
    bn_running_var = self.bn.running_var
    bn_num_batches_tracked = self.bn.num_batches_tracked
    conv2d = torch.ops.aten.conv2d.default(x, conv_weight, conv_bias);  x = conv_weight = conv_bias = None
    add = torch.ops.aten.add.Tensor(bn_num_batches_tracked, 1)
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, True, 0.1, 1e-05);  conv2d = bn_weight = bn_bias = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    copy__default = torch.ops.aten.copy_.default(bn_running_mean, getitem_3);  bn_running_mean = getitem_3 = None
    copy__default_1 = torch.ops.aten.copy_.default(bn_running_var, getitem_4);  bn_running_var = getitem_4 = None
    copy__default_2 = torch.ops.aten.copy_.default(bn_num_batches_tracked, add);  bn_num_batches_tracked = add = None
    return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )

    def test_constrain_size_in_eager(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                n = x.max().item()
                torch._check_is_size(n)
                return y + n

        fn = Module()
        ep = export(
            fn,
            (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3))),
        )
        test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
        self.assertTrue(torch.allclose(ep.module()(*test_inp), fn(*test_inp)))

    def test_constrain_size_with_constrain_value(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                n = x.max().item()
                torch._check(n >= 2)
                torch._check(n <= 10)
                torch._check_is_size(n)
                return y + n

        fn = Module()
        with self.assertRaisesRegex(
            RuntimeError, r"Expected cond to be True, but got False"
        ):
            _ = fn(torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))

        ep = export(
            fn,
            (torch.randint(3, 4, (2, 2)), torch.randint(3, 5, (2, 3))),
        )
        with self.assertRaisesRegex(RuntimeError, "Invalid value range for 1 between"):
            test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
            _ = ep.module()(*test_inp)

    def test_constrain_size_with_various_cases(self):
        class Module1(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 0)
                return y.sum() + torch.ones(n, 5).sum()

        case1 = Module1()

        class Module2(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 0)
                torch._check(n <= 6)
                return y.sum() + torch.ones(n, 5).sum()

        case2 = Module2()

        class Module3(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 0)
                torch._check(n <= 1)
                return y.sum() + torch.ones(n, 5).sum()

        case3 = Module3()

        class Module4(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 2)
                return y.sum() + torch.ones(n, 5).sum()

        case4 = Module4()

        class Module5(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 1)
                return y.sum() + torch.ones(n, 5).sum()

        case5 = Module5()

        ep = export(case1, (torch.tensor(1), torch.ones(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError, r"Expected cond to be True, but got False"
        ):
            _ = case1(torch.tensor(-1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(1), torch.ones(4, 5)),
                case1(torch.tensor(1), torch.ones(4, 5)),
            )
        )

        ep = export(case2, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case2(torch.tensor(7), torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case2(torch.tensor(9), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case2(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        _ = case3(torch.tensor(1), torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case4(torch.tensor(1), torch.randn(4, 5))

        ep = export(case4, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case4(torch.tensor(1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case4(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        ep = export(case5, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case5(torch.tensor(0), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case5(torch.tensor(5), torch.ones(4, 5)),
            )
        )

    def test_automatic_constrain_size(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                return y.sum() + torch.ones(n, 5).sum()

        ep = export(M(), (torch.tensor(1), torch.ones(4, 5)))

        # This is because we insert sym_constrain_range in the graph now
        error_msg = r"Invalid value range for -1 between"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            _ = ep.module()(torch.tensor(-1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(1), torch.ones(4, 5)),
                M()(torch.tensor(1), torch.ones(4, 5)),
            )
        )

    def test_constrain_decomp(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.freq = torch.ones(5, 5)

            def forward(self, start_pos: torch.Tensor):
                pos = start_pos.item()
                torch._check_is_size(pos)
                torch._check(pos >= 0)
                torch._check(pos <= 4)
                return self.freq[pos] * self.freq[pos]

        ep = torch.export.export(M(), (torch.tensor(1),))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        decompose_ep = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(decompose_ep.graph_module.code)

    def test_mixed_input(self):
        class Module(torch.nn.Module):
            def forward(self, a, b, alpha: int):
                return torch.add(a, b, alpha=alpha)

        func = Module()

        a = torch.rand(1, 2)
        b = torch.rand(1, 2)
        alpha = 10

        exported = export(func, (a, b, alpha))
        for node in exported.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(isinstance(node.meta["val"], (Tensor, int)))

    def test_export_with_inline_constraints(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                torch._check(a >= 4)
                torch._check(a <= 7)
                return torch.empty((a, 4))

        f = Module()
        ep = export(f, (torch.tensor([5]),))
        self.assertEqual(ep.module()(torch.tensor([6])).shape, (6, 4))

        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Invalid value range for 30 between \[4, 7\]",
        ) as cm:
            ep.module()(torch.tensor([30]))

    def test_export_with_inline_constraints_complex(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                torch._check(a >= 4)
                torch._check(a <= 7)
                empty = torch.empty((a, 4))

                return torch.cat((empty.transpose(0, 1), torch.zeros(6, a)), 0)

        f = Module()
        ep = export(f, (torch.tensor([6]),))
        self.assertEqual(ep.module()(torch.tensor([5])).shape, (10, 5))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

    def test_to_module_with_mutated_buffer(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.ones(1), buffer))

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.tensor(2, dtype=torch.float), buffer))

    def test_to_module_with_mutated_buffer_multiple(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                self.bar.buf.add_(2)
                bar = self.bar(x)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )

    def test_runtime_assert_for_prim(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        foo = Foo()
        tensor_inp = torch.ones(7, 5)
        dim0_x = torch.export.Dim("dim0_x", min=6)
        dynamic_shapes = {"x": {0: dim0_x}, "y": None}
        exported = torch.export.export(
            foo, (tensor_inp, 5), dynamic_shapes=dynamic_shapes
        )
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(8, 5), 5), foo(torch.ones(8, 5), 5)
            )
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5, but got 6"),
        ):
            _ = exported.module()(torch.ones(8, 5), 6)

        exported = torch.export.export(
            foo, (tensor_inp, 5.0), dynamic_shapes=dynamic_shapes
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5.0, but got 6.0"),
        ):
            _ = exported.module()(torch.ones(7, 5), 6.0)

    def test_runtime_assert_for_prm_str(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b, mode):
                return torch.div(a, b, rounding_mode=mode)

        foo = Foo()
        inps = (torch.randn(4, 4), torch.randn(4), "trunc")
        exported = export(foo, inps)
        with self.assertRaisesRegex(
            RuntimeError, "to be equal to trunc, but got floor"
        ):
            _ = exported.module()(torch.randn(4, 4), torch.randn(4), "floor")
        self.assertTrue(torch.allclose(exported.module()(*inps), foo(*inps)))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193701923
    def test_to_module_with_mutated_buffer_multiple_update_sub_later(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )

    def test_retracable_ep(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        inp = torch.ones(5, 5)
        exported = torch.export.export(Foo(), (inp,))
        reexported = torch.export.export(exported.module(), (inp,))

        self.assertTrue(torch.allclose(Foo()(inp), reexported.module()(inp)))

        dim0_x = torch.export.Dim("dim0_x")
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes=({0: dim0_x},))
        reexported = torch.export.export(exported.module(), (inp,))
        with self.assertRaisesRegex(
            RuntimeError, "shape\[0\] to be equal to 5, but got 7"
        ):
            reexported.module()(torch.ones(7, 5))

        reexported = torch.export.export(
            exported.module(), (inp,), dynamic_shapes=({0: dim0_x},)
        )
        self.assertTrue(
            torch.allclose(
                Foo()(torch.ones(7, 5)), reexported.module()(torch.ones(7, 5))
            )
        )

        # can't retrace with invalid inputs with respect to the original ExportedProgram
        dim0_x_v2 = torch.export.Dim("dim0_x_v2", min=3)
        exported_v2 = torch.export.export(
            Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x_v2}}
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be >= 3, but got 2"),
        ):
            torch.export.export(exported_v2.module(), (torch.randn(2, 2),))

    def test_export_cond(self):
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self):
                return self.buffer.cos()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()

            def forward(self, x):
                def true_fn(x):
                    return x.cos() + self.a().sum()

                def false_fn(x):
                    return x.sin()

                return cond(x.shape[0] > 4, true_fn, false_fn, [x])

        inp = torch.ones(6, 4)
        ep = export(
            Foo(),
            (inp,),
        )
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(6, 4)), Foo()(torch.ones(6, 4)))
        )

    def test_aten_lift_fresh_copy(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.lift_fresh_copy(x)

        ep = export(M(), (torch.ones(6, 4),))
        found = False

        op = "torch.ops.aten.clone.default"
        FileCheck().check_count(op, 1, exactly=True).run(ep.graph_module.code)

    def test_cond_buffers(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                self.register_buffer("buffer", torch.ones(2, 3) + 1)

            def true_fn(self, x):
                return x + self.param

            def false_fn(self, x):
                return x + self.buffer

            def forward(self, x):
                return cond(x.shape[0] == 4, self.true_fn, self.false_fn, [x])

        inp = torch.ones(2, 3)
        ep = torch.export.export(M(), (inp,))
        inp = torch.randn(2, 3)
        epm = ep.module()
        self.assertTrue(torch.allclose(epm(inp), M()(inp)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 1
            )

    # map_fn references module outside the module hierarchy
    @unittest.expectedFailure
    def test_map_buffers(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.tensor(5), requires_grad=False)
                )
                self.register_buffer("buffer", torch.tensor(6) + 1)

        m1 = M1()

        def map_fn(x, y):
            z = x + y + m1.param + m1.buffer
            z.add_(4)
            return z

        class M(torch.nn.Module):
            def forward(self, xs, y):
                return map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2), torch.tensor(3))
        ep = torch.export.export(M(), example_inputs)
        example_inputs = (torch.randn(3, 2), torch.tensor(3))
        epm = ep.module()
        self.assertTrue(torch.allclose(epm(*example_inputs), M()(*example_inputs)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 2
            )

    @testing.expectedFailureSerDer  # We don't preserve metadata on graph module
    @testing.expectedFailureNonStrict
    def test_retrace_graph_level_meta_preservation(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                if x.shape[0] > 4:
                    return x.cos()
                return x.sin()

        inp = torch.ones(7, 5)
        dim0_x = torch.export.Dim("dim0_x", min=6)
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x}})
        stateful_module = exported.module()
        self.assertTrue(len(stateful_module.meta["input_shape_constraints"]), 1)

        re_exported = export(stateful_module, (inp,), dynamic_shapes=({0: dim0_x},))
        self.assertTrue(
            len(re_exported.graph_module.meta["input_shape_constraints"]) == 1
        )
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(7, 5)),
                re_exported.module()(torch.ones(7, 5)),
            )
        )

        re_exported_v2 = export(exported.module(), (inp,))
        self.assertTrue(
            len(re_exported_v2.graph_module.meta["input_shape_constraints"]) == 0
        )
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(7, 5)),
                re_exported_v2.module()(torch.ones(7, 5)),
            )
        )

    def test_check_is_size_error(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                # We cannot automatically infer a is a size here because view
                # accepts -1
                return torch.randn(24).view(a, 4)

        f = Module()
        if is_non_strict_test(self._testMethodName):
            error = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
            error_msg = r"Could not guard on data-dependent expression"
        else:
            error = torch._dynamo.exc.UserError
            error_msg = (
                r"Tried to use data-dependent value in the subsequent computation"
            )
        with self.assertRaisesRegex(error, error_msg):
            _ = export(f, (torch.tensor(6),))

    def test_train_eval_on_exported_preautograd_module(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                if x.shape[0] > 4:
                    return x.cos()
                return x.sin()

        graph_module = _export(Foo(), (torch.ones(7, 5),), pre_dispatch=True).module()
        with self.assertRaisesRegex(
            NotImplementedError, r"Calling train\(\) is not supported yet."
        ):
            graph_module.train()

        with self.assertRaisesRegex(
            NotImplementedError, r"Calling eval\(\) is not supported yet."
        ):
            graph_module.eval()

    # TODO Retracing a module with constant attrs don't work.(T193692674)
    @testing.expectedFailureTrainingIRToRunDecomp
    @testing.expectedFailureRetraceability  # T183144788
    def test_lifted_constants(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor(3)

        f = Module()
        ep = export(f, (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 2)
        self.assertEqual(len(ep.constants), 1)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Foo(), (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 3)

        inp = (torch.tensor(5),)
        self.assertTrue(torch.allclose(ep.module()(*inp), Foo()(*inp)))

        transform = ep.run_decompositions()
        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertTrue(torch.allclose(ep.module()(*inp), transform.module()(*inp)))

    @testing.expectedFailureRetraceability  # T183144788
    @testing.expectedFailureTrainingIRToRunDecomp  # T193701164
    def test_tensor_attribute_zero_args(self):
        class Foo(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.x = torch.tensor(value)

            def forward(self):
                return self.x.clone()

        m = Foo([1, 2])
        ep = export(m, ())
        self.assertEqual(ep.graph_signature.lifted_tensor_constants, ["x"])

    def test_preserve_shape_dynamism_for_unused_inputs(self):
        @dataclass
        class Input:
            f: torch.Tensor
            p: torch.Tensor

        torch._export.utils.register_dataclass_as_pytree_node(
            Input,
            serialized_type_name="test_preserve_shape_dynamism_for_unused_inputs.Input",
        )

        class Module(torch.nn.Module):
            def forward(self, x: Input):
                return x.f + 1

        mod = Module()
        example_inputs = (Input(f=torch.ones(10, 4), p=torch.zeros(10, 4)),)
        ep_static = torch.export.export(mod, example_inputs)
        for node in ep_static.graph.nodes:
            if node.op == "placeholder":
                for s in node.meta["val"].shape:
                    self.assertIsInstance(s, int)

        dim0_x_f, dim0_x_p = torch.export.dims("dim0_x_f", "dim0_x_p")
        dynamic_shapes = {"x": [{0: dim0_x_f}, {0: dim0_x_p}]}
        ep_dynamic = torch.export.export(
            mod, example_inputs, dynamic_shapes=dynamic_shapes
        )
        for node in ep_dynamic.graph.nodes:
            if node.op == "placeholder":
                for i, s in enumerate(node.meta["val"].shape):
                    if i == 0:
                        self.assertIsInstance(s, torch.SymInt)
                    else:
                        self.assertIsInstance(s, int)

    def test_multiple_definitions_same_name_dim(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        A = torch.export.Dim("C", min=3)
        B = torch.export.Dim("C", max=12)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Found different definitions Dim\\(.*min=3\\) and Dim\\(.*max=12\\) "
            "for the same symbolic dimension",
        ):
            torch.export.export(
                Foo(),
                (torch.randn(10, 10), torch.randn(10, 10)),
                dynamic_shapes={"x": (A, B), "y": (B, A)},
            )

    def test_export_with_wrong_inputs(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        exported_program = export(MyModule(), (torch.rand(2, 3),), {})
        with self.assertRaisesRegex(ValueError, "Trying to flatten user inputs"):
            exported_program.module()(torch.rand(2, 3), torch.rand(2, 3))

    def test_export_decomps_simple(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp)
        state_dict = ep.state_dict

        self.assertTrue(torch.allclose(ep.module()(*inp), m(*inp)))

        core_aten_ep = ep.run_decompositions()
        FileCheck().check_count("torch.ops.aten.permute.default", 1, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        FileCheck().check_count("torch.ops.aten.t.default", 0, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        self.assertTrue(torch.allclose(core_aten_ep.module()(*inp), m(*inp)))
        self.assertEqual(id(state_dict), id(ep.state_dict))

    def test_export_decomps_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp, dynamic_shapes={"x": {0: Dim("batch")}})

        core_aten_ep = ep.run_decompositions()

        input_node = [
            node for node in core_aten_ep.graph.nodes if node.op == "placeholder"
        ][-1]
        self.assertTrue(isinstance(input_node.meta["val"].shape[0], torch.SymInt))

        FileCheck().check_count("torch.ops.aten.permute.default", 1, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        FileCheck().check_count("torch.ops.aten.t.default", 0, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        self.assertTrue(torch.allclose(core_aten_ep.module()(*inp), m(*inp)))

    def test_nonzero_2(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.nonzero(x)

        f = Module()
        ep = export(f, (torch.ones(2),))
        inp = torch.randn(2)
        self.assertTrue(torch.allclose(ep.module()(inp), torch.nonzero(inp)))

    def test_redundant_asserts(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = x.item()
                torch._check_is_size(y)
                return torch.zeros(y)

        f = Foo()

        ep = export(f, (torch.tensor([3]),))

        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 1, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

        ep = ep.run_decompositions()

        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 1, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_non_arg_name_dynamic_shapes_api(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return a.sum() + b.sum()

        foo = Foo()
        dim = torch.export.Dim("dim")
        ep = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=(None, {0: dim}),
        )

        test_inp = (torch.randn(4, 4), torch.randn(7, 4))
        self.assertEqual(ep.module()(*test_inp), foo(*test_inp))

        ep_v2 = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=(None, None),
        )
        with self.assertRaisesRegex(
            RuntimeError, "shape\[0\] to be equal to 4, but got 7"
        ):
            ep_v2.module()(*test_inp)

    def test_constant_output(self):
        class ModuleConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.randn(3, 2)

            def forward(self):
                return self.b

        class ModuleNestedConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bff = torch.randn(3, 2)

            def forward(self, x, y):
                return {"prediction": (x + y, self.bff)}

        mod = ModuleConstant()
        ep = torch.export.export(mod, ())
        self.assertEqual(ep.module()(), mod())

        args = (torch.randn(3, 2), torch.randn(3, 2))
        mod = ModuleNestedConstant()
        ep = torch.export.export(mod, args)
        self.assertEqual(ep.module()(*args), mod(*args))

    def test_non_arg_name_dynamic_shapes_api_with_kwarg(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b, kw1, kw2):
                return a.sum() + b.sum() + kw1.sum() - kw2.sum()

        foo = Foo()
        dim = torch.export.Dim("dim")
        dim_for_kw1 = torch.export.Dim("dim_for_kw1")
        ep = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            {"kw2": torch.ones(4, 4), "kw1": torch.zeros(4, 4)},
            # We are specifying dynamism on the first kwarg even though user passed in
            # different order
            dynamic_shapes=(None, {0: dim}, {0: dim_for_kw1}, None),
        )

        test_inp = (torch.randn(4, 4), torch.randn(7, 4))
        test_kwargs = {"kw2": torch.ones(4, 4), "kw1": torch.zeros(9, 4)}
        # This should work even if the kwarg order are flipped.
        self.assertEqual(
            ep.module()(*test_inp, **test_kwargs), foo(*test_inp, **test_kwargs)
        )

    def test_non_arg_name_dynamic_shapes_api_with_container_type(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return a[0].sum() + a[1].sum() + b.sum()

        inp_a = (torch.randn(4, 4), torch.randn(4, 4))
        inp_b = torch.randn(4, 4)
        inp = (inp_a, inp_b)

        count = 0

        def dynamify_inp(x):
            # Mark the second input a[1] dynamic
            nonlocal count
            if count == 1:
                dim = torch.export.Dim("dim", min=3)
                count += 1
                return {0: dim}
            count += 1
            return None

        dynamic_shapes = tree_map(dynamify_inp, inp)

        foo = Foo()
        ep = torch.export.export(foo, inp, dynamic_shapes=dynamic_shapes)

        test_inp = ((torch.randn(4, 4), torch.randn(2, 4)), torch.randn(4, 4))
        with self.assertRaisesRegex(RuntimeError, "shape\[0\] to be >= 3, but got 2"):
            ep.module()(*test_inp)

    def test_nested_module(self):
        class M1(torch.nn.Module):
            def forward(self, x):
                return x + x

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        ep = export(M2(), inps)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        add_nodes = [
            node
            for node in ep.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor
        ]
        self.assertEqual(len(add_nodes), 1)
        add_node = add_nodes[0]
        self.assertEqual(len(add_node.meta["nn_module_stack"]), 1)
        self.assertTrue("M2" in list(add_node.meta["nn_module_stack"].values())[0][1])

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=2] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %x), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    def test_nested_module_with_init_buffer(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.ones(3, 3)

            def forward(self, x):
                return x + self.b

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        ep = export(M2(), inps)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 0)

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=2] = placeholder[target=x]
    %ones : [num_users=1] = call_function[target=torch.ops.aten.ones.default](args = ([3, 3],), kwargs = {device: cpu, pin_memory: False})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %ones), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    @testing.expectedFailureRetraceability  # Retracing tensor constants results in buffers
    @testing.expectedFailureTrainingIRToRunDecomp  # T193692674
    def test_nested_module_with_constant_buffer(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.tensor(5)

            def forward(self, x):
                return x + self.b

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        ep = export(M2(), inps)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 1)

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%lift_fresh_copy,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %detach), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    def test_nested_module_with_parameter(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(3, 3))
                self.b = torch.nn.Parameter(torch.tensor(5.0))

            def forward(self, x):
                return x + self.a * self.b

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        # Strict export segfaults (Issue #128109)
        ep = torch.export.export(M2(), inps, strict=False)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 1)

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %ones : [num_users=1] = call_function[target=torch.ops.aten.ones.default](args = ([3, 3],), kwargs = {device: cpu, pin_memory: False})
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%ones,), kwargs = {})
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %detach_1 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%lift_fresh_copy,), kwargs = {})
    %detach_2 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach_1,), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%detach, %detach_2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul_1,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    def test_lazy_module_kwargs(self):
        class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def initialize_parameters(self, *args, **kwargs):
                pass

            def forward(self, x, y):
                return x + y

        m = LazyModule()
        ep = torch.export.export(
            m, (), {"x": torch.randn(3, 3), "y": torch.randn(3, 3)}
        )
        inputs = {"x": torch.randn(3, 3), "y": torch.randn(3, 3)}
        self.assertEqual(ep.module()(**inputs), m(**inputs))

    def test_retrace_pre_autograd(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(4, 4))

            def forward(self, x):
                self.buffer.add_(4)
                return x.sum() + self.buffer.sum()

        inp = torch.randn(4, 4)
        gm = _export(
            Foo(),
            (inp,),
            dynamic_shapes=({0: torch.export.Dim("dim", min=3)},),
            pre_dispatch=True,
        ).module()

        with self.assertRaisesRegex(
            RuntimeError, escape("Expected input at *args[0].shape[0]")
        ):
            gm(torch.randn(2, 2))

        with self.assertRaisesRegex(
            RuntimeError, escape("Expected input at *args[0].shape[0]")
        ):
            torch.export.export(gm, (torch.randn(2, 2),))

        ep = torch.export.export(
            gm,
            (torch.randn(5, 4),),
            dynamic_shapes=({0: torch.export.Dim("dim", min=3)},),
        )

        test_inp = torch.ones(8, 4)
        self.assertTrue(torch.allclose(ep.module()(test_inp), Foo().forward(test_inp)))

    def test_runtime_assert_with_size(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                a = x.item()
                torch._check_is_size(a)
                torch._check(a <= y.size(0))
                return y[:a]

        ep = export(
            M(),
            (torch.tensor(5), torch.ones(10)),
            dynamic_shapes={"x": None, "y": {0: torch.export.Dim("t")}},
        )
        inp = (torch.tensor(6), torch.randn(13))
        self.assertTrue(torch.allclose(ep.module()(*inp), M()(*inp)))

    # TODO Retracing a module with constant attrs don't work.(T193692674)
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_issue_113041(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor(1.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.a

        def forward_hook(module: torch.nn.Module, inputs, output) -> torch.Tensor:
            return 2 * output

        seq = torch.nn.Sequential(TestModule()).eval()
        seq.b = torch.tensor(2)
        handle = seq.register_forward_hook(forward_hook)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = seq

            def forward(self, x):
                return self.seq(x) + self.seq.b

        inp = (torch.randn(2, 8),)
        ep = export(M(), inp)  # This errors because dynamo adds an extra input

    def test_export_with_fake_tensor_inputs(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        # Put the inputs on a device
        with fake_mode, torch.device("meta"):
            x = torch.rand(5, 2, 2)
            model = Model()

            exported_program = torch.export.export(model, (x,))
            export_res = exported_program.module()(x)
            exp_res = model(x)
            all_meta_val = [
                node.meta["val"]
                for node in exported_program.graph_module.graph.nodes
                if "val" in node.meta
            ]
            self.assertTrue(export_res.size() == exp_res.size())
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            self.assertTrue(
                all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val)
            )
            decomposed_ep = exported_program.run_decompositions()
            export_res = decomposed_ep.module()(x)
            self.assertTrue(export_res.size() == exp_res.size())

    def test_export_with_fake_tensor_inputs_on_cuda_devices(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        # Put the inputs on a device
        with fake_mode, torch.device("meta"):
            x = torch.rand(5, 2, 2)
            model = Model()

        # Manualy set the fake_device of fake tensors.
        x.fake_device = torch.device("cuda:0")
        for n, p in model.named_parameters():
            p.fake_device = torch.device("cuda:0")

        # Need to set all the requires_grad of tensors to False, because fake_tensor with CUDA device
        # doesn't quite work well with aot_autograd right now due to some logic fails
        # the check in call getDeviceGuardImpl in InputMetadata.
        x.requires_grad = False
        for n, p in model.named_parameters():
            p.requires_grad = False

        def check_device_and_fake_mode():
            exported_program = torch.export.export(model, (x,))
            export_res = exported_program.module()(x)
            exp_res = model(x)
            all_meta_val = [
                node.meta["val"]
                for node in exported_program.graph_module.graph.nodes
                if "val" in node.meta
            ]
            self.assertTrue(export_res.size() == exp_res.size())
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            self.assertTrue(
                all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val)
            )

        check_device_and_fake_mode()

    def test_run_decomposition_supports_user_input_mutation(self):
        class SingleOp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.ops.aten.native_batch_norm

            def forward(
                self,
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
                **kwargs,
            ):
                return self.op(
                    input,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    training,
                    momentum,
                    eps,
                    **kwargs,
                )

        input = torch.randn(5, 5, 5)
        weight = torch.randn(5)
        bias = torch.randn(5)
        running_mean = torch.randn(5)
        running_var = torch.randn(5)
        training = True
        momentum = 0.5
        eps = 0.6

        model = SingleOp()
        output = model(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )

        ep = torch.export.export(
            model,
            args=(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            ),
        )
        ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)
        self.assertEqual(
            ep.module()(
                input, weight, bias, running_mean, running_var, training, momentum, eps
            ),
            output,
        )

    def test_export_graph_with_no_inputs(self):
        # We saw this pattern when users want to export
        # a graph that initlizes the states of a model.
        class Module(torch.nn.Module):
            def forward(self):
                return torch.randn(3, 4), torch.randn(3, 4)

        f = Module()
        ep = torch.export.export(f, ())
        a, b = ep.module()()
        self.assertEqual(a.size(), torch.Size([3, 4]))
        self.assertEqual(b.size(), torch.Size([3, 4]))

    def test_pad_sequence(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch._C._nn.pad_sequence([x])

        m0 = Module()
        inputs = (torch.randn(3, 2),)
        ep = torch.export.export(
            m0, inputs, dynamic_shapes={"x": {0: Dim("batch_size")}}
        )
        self.assertEqual(ep.module()(*inputs), m0(*inputs))

        class ModuleBatchFirst(torch.nn.Module):
            def forward(self, x):
                return torch._C._nn.pad_sequence([x], batch_first=True)

        m1 = ModuleBatchFirst()
        inputs = (torch.randn(3, 2),)
        ep = torch.export.export(
            m1, inputs, dynamic_shapes={"x": {0: Dim("batch_size")}}
        )
        self.assertEqual(ep.module()(*inputs), m1(*inputs))

        class ModuleMulti(torch.nn.Module):
            def forward(self, x, y, z):
                return torch._C._nn.pad_sequence([x, y, z])

        m2 = ModuleMulti()
        inputs = (torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2))
        ep = torch.export.export(
            m2,
            inputs,
            dynamic_shapes={
                "x": {0: Dim("batch_size")},
                "y": {0: Dim("y")},
                "z": {0: Dim("z")},
            },
        )
        self.assertEqual(ep.module()(*inputs), m2(*inputs))

        class ModuleMultiBatchFirst(torch.nn.Module):
            def forward(self, x, y, z):
                return torch._C._nn.pad_sequence([x, y, z], batch_first=True)

        m3 = ModuleMulti()
        inputs = (torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2))
        ep = torch.export.export(
            m2,
            inputs,
            dynamic_shapes={
                "x": {0: Dim("batch_size")},
                "y": {0: Dim("y")},
                "z": {0: Dim("z")},
            },
        )
        self.assertEqual(ep.module()(*inputs), m3(*inputs))

    def test_export_then_compile_tensor_ctor(self):
        class M(torch.nn.Module):
            def forward(self, scores, mask):
                scores = scores.masked_fill(
                    mask, torch.tensor(torch.finfo(scores.dtype).min)
                )  # (bs, n_heads, q_length, k_length)
                return scores

        tensor_cpu = torch.randn(2, 4)
        mask_cpu = torch.BoolTensor(
            [[False, True, False, False], [False, False, False, False]]
        )

        m = M().eval()
        # res_ref = m(tensor_cpu, mask_cpu)
        # print("res_ref is: {}".format(res_ref), flush=True)

        exported_model = _export(m, (tensor_cpu, mask_cpu), pre_dispatch=True).module()
        optimized_model = torch.compile(exported_model)
        optimized_model(tensor_cpu, mask_cpu)

    @testing.expectedFailureTrainingIRToRunDecomp  # T193701923
    def test_export_input_mutation_static_shape(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x, y):
                x.view(3, 2, -1).add_(y)
                return x

        inputs = (torch.randn(12), torch.tensor(2))
        model = MutationModel()
        ep = export(model, inputs)
        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        self.assertEqual(ep.module()(*inputs_export), model(*inputs_model))
        self.assertEqual(inputs[0] + torch.tensor(2), inputs_model[0])
        self.assertEqual(inputs[0] + torch.tensor(2), inputs_export[0])

    def test_export_input_mutation_dynamic_shape(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x, y):
                x[0].mul_(y)
                return x

        inputs = ((torch.randn(12), torch.randn(3, 2)), 2.0)
        model = MutationModel()
        ep = torch.export.export(
            model,
            inputs,
            dynamic_shapes={"x": ({0: torch.export.Dim("dim")}, None), "y": None},
        )
        nodes = list(ep.graph.nodes)
        self.assertEqual(nodes[0].op, "placeholder")
        self.assertIsInstance(nodes[0].meta["val"], torch.Tensor)
        self.assertIsInstance(nodes[0].meta["val"].shape[0], torch.SymInt)

        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        self.assertEqual(ep.module()(*inputs_export), model(*inputs_model))
        self.assertEqual(inputs[0][0] * 2.0, inputs_model[0][0])
        self.assertEqual(inputs[0][0] * 2.0, inputs_export[0][0])

    def test_export_input_mutation_bug(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x[:, :2, :] = x[:, :2, :] + 1
                return x

        inputs = (torch.ones(4, 4, 4),)
        ep = torch.export.export(M(), inputs)
        m = ep.module()

        # Make the name conflict with a placeholder name that we get from
        # aot_export
        for i, node in enumerate(m.graph.nodes):
            if node.op == "placeholder":
                node.name = f"arg0_{i + 1}"
        m.recompile()

        ep = torch.export.export(m, inputs)

        inputs = (torch.randn(4, 4, 4),)
        self.assertEqual(
            ep.module()(*copy.deepcopy(inputs)), M()(*copy.deepcopy(inputs))
        )

    def test__scaled_dot_product_flash_attention(self):
        class Module(torch.nn.Module):
            def forward(self, q, k, v):
                res = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                return res[0]

        m = Module()
        inputs = (
            torch.randn(5, 4, 3, 2),
            torch.randn(5, 4, 3, 2),
            torch.randn(5, 4, 3, 2),
        )
        ep = export(m, inputs)
        self.assertEqual(ep.module()(*inputs), m(*inputs))

    @testing.expectedFailureSerDer  # symfloat nyi
    def test_sym_sqrt(self):
        import math

        class M(torch.nn.Module):
            def forward(self, x):
                return x / torch.sym_sqrt(x.shape[0])

        ep = export(M(), (torch.ones(16, 4),), dynamic_shapes={"x": {0: Dim("dim")}})
        _ExportPassBaseDeprecatedDoNotUse()(ep.graph_module)
        FileCheck().check_count("torch._sym_sqrt", 1, exactly=True).run(
            ep.graph_module.code
        )

    def test_check_specialized_int(self):
        class SingleOp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.ops.aten.scatter_add

            def forward(self, t, dim, index, src, **kwargs):
                return self.op(t, dim, index, src, **kwargs)

        t = torch.randn(10, 5)
        dim = -1
        index = torch.tensor(
            [
                [2, 4, 3, 1, 0],
                [0, 2, 1, 4, 3],
                [3, 1, 4, 2, 0],
                [4, 0, 3, 1, 2],
                [3, 0, 4, 1, 2],
            ]
        )
        src = torch.randn(5, 5)

        model = SingleOp()
        output = model(t, dim, index, src)

        ep = torch.export.export(model, args=(t, dim, index, src))
        ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)
        self.assertEqual(ep.module()(t, dim, index, src), output)

    def test_fqn(self):
        class NestedChild(torch.nn.Module):
            def forward(self, x):
                return x / x

        class Child1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("child2buffer", torch.ones(2, 3))

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        test_inp = torch.randn(2, 3)

        torch_gm = _export_to_torch_ir(orig_eager, (torch.rand(2, 3),), {})
        for k, v in orig_eager.state_dict().items():
            normalized_k = k.replace(".", "_")
            self.assertIn(normalized_k, torch_gm.state_dict())
            self.assertEqual(v, torch_gm.state_dict()[normalized_k])
        self.assertTrue(torch.allclose(torch_gm(test_inp), orig_eager(test_inp)))

        pre_autograd_gm = torch.export._trace._export(
            orig_eager, (torch.rand(2, 3),), {}, pre_dispatch=True
        ).module()
        for k, v in orig_eager.state_dict().items():
            normalized_k = k.replace(".", "_")
            self.assertIn(k, pre_autograd_gm.state_dict())
            self.assertEqual(v, pre_autograd_gm.state_dict()[k])
        self.assertTrue(torch.allclose(pre_autograd_gm(test_inp), orig_eager(test_inp)))

        ep = export(orig_eager, (torch.rand(2, 3),), {})
        for k, v in orig_eager.state_dict().items():
            # We do not need to normalize the key here because exported
            # program's state dict is able to contain the module information.
            self.assertIn(k, ep.state_dict)
            self.assertEqual(v, ep.state_dict[k])
        self.assertTrue(torch.allclose(ep.module()(test_inp), orig_eager(test_inp)))

    def test_nn_module_stack(self):
        class Leaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()
                self.register_buffer("buffer", torch.randn(4, 4))

            def forward(self, x):
                return self.buffer.sum() + self.leaf(x).sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                y = self.bar.buffer + x
                return (self.bar(x) + y.sum(),)

        inp = (torch.randn(4, 4),)
        mod = Foo()
        ep_strict = torch.export.export(mod, inp).run_decompositions()
        ep_non_strict = torch.export.export(mod, inp, strict=False).run_decompositions()

        gm_unflat_non_strict = unflatten(ep_non_strict)
        self.assertTrue(hasattr(gm_unflat_non_strict, "bar"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "buffer"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "leaf"))

        gm_unflat_strict = unflatten(ep_strict)

        self.assertEqual(gm_unflat_non_strict(*inp), gm_unflat_strict(*inp))
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar.leaf.linear.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %weight : [num_users=1] = get_attr[target=weight]
    %bias : [num_users=1] = get_attr[target=bias]
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%weight, [1, 0]), kwargs = {})
    %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%bias, %x, %permute), kwargs = {})
    return addmm""",
        )

        gm_flat_non_strict = ep_non_strict.module()
        gm_flat_strict = ep_strict.module()

        self.assertEqual(gm_flat_non_strict(*inp), gm_flat_strict(*inp))

    def test_nn_module_stack_shared_submodule(self):
        class Leaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()
                self.register_buffer("buffer", torch.randn(4, 4))

            def forward(self, x):
                return self.buffer.sum() + self.leaf(x).sum()

        class BarDifferent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()

            def forward(self, x):
                a = self.leaf(x).sum()
                b = self.leaf(x).sum()
                return a + b

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()
                self.bar_different = BarDifferent()

            def forward(self, x):
                y = self.bar.buffer + x
                return (
                    self.bar(x) + self.bar_different(x + 2),
                    y.sum(),
                )

        inp = (torch.randn(4, 4),)
        mod = Foo()
        ep_strict = torch.export.export(mod, inp)
        ep_non_strict = torch.export.export(mod, inp, strict=False)

        gm_unflat_non_strict = unflatten(ep_non_strict)
        self.assertTrue(hasattr(gm_unflat_non_strict, "bar"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "buffer"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "leaf"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar_different, "leaf"))

        gm_unflat_strict = unflatten(ep_strict)

        self.assertEqual(gm_unflat_non_strict(*inp), gm_unflat_strict(*inp))
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar.leaf.linear.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %weight : [num_users=1] = get_attr[target=weight]
    %bias : [num_users=1] = get_attr[target=bias]
    %linear : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%x, %weight, %bias), kwargs = {})
    return linear""",
        )
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar_different.leaf.linear.graph).strip(),
            """\
graph():
    %add_2 : [num_users=1] = placeholder[target=add_2]
    %weight : [num_users=1] = get_attr[target=weight]
    %bias : [num_users=1] = get_attr[target=bias]
    %linear_1 : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%add_2, %weight, %bias), kwargs = {})
    return linear_1""",
        )

        gm_flat_non_strict = ep_non_strict.module()
        gm_flat_strict = ep_strict.module()

        self.assertEqual(gm_flat_non_strict(*inp), gm_flat_strict(*inp))

    def test_stack_trace(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                x = self.linear(x)
                x *= 2.0
                return x

        ep = export(
            Foo(),
            (torch.randn(4, 4),),
        )
        # check correct lines are in stack trace
        trace_mul = [node for node in ep.graph.nodes if node.name == "mul"][0].meta.get(
            "stack_trace", ""
        )
        self.assertTrue(
            re.search(r"test_export.py.*in forward\n.*x \*= 2.0", trace_mul)
        )
        trace_addmm = [
            node for node in ep.graph.nodes if node.name in ["addmm", "linear"]
        ][0].meta.get("stack_trace", "")
        self.assertTrue(
            re.search(
                r"test_export.py.*in forward\n.*x = self.linear\(x\)", trace_addmm
            )
        )

    @testing.expectedFailureTrainingIRToRunDecomp  # T193702033
    def test_sym_stack_trace(self):
        # TODO(avik): update this test with torch._check*
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                y = torch.sym_constrain_range_for_size(y.item(), min=2)
                z = x.shape[0] == 4
                z = torch.sym_ite(z, x.shape[0], x.shape[1])
                return z

        ep = export(
            Foo(),
            (torch.randn(4, 4), torch.tensor(5)),
            dynamic_shapes={"x": (Dim("dx0"), Dim("dx1")), "y": None},
        )
        # stack trace for sym call constrain_range
        trace_constrain_range = [  # different names for serdes/pre-dispatch
            node
            for node in ep.graph.nodes
            if node.name
            in ["sym_constrain_range_for_size", "sym_constrain_range_for_size_default"]
        ][0].meta.get("stack_trace", None)
        self.assertTrue(
            re.search(
                r"in forward\n.*torch.sym_constrain_range_for_size",
                trace_constrain_range,
            )
        )

    def test_cond_with_module_stack_export_with(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_fn(x):
                    return self.linear(x).cos()

                def false_fn(x):
                    return self.linear(x).sin()

                return torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])

        class CondExport(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x.cos() + self.bar(x)

        inp = (torch.randn(4, 4),)
        ep = torch.export.export(CondExport(), inp, strict=False)
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, p_bar_linear_weight, p_bar_linear_bias, x):
    cos = torch.ops.aten.cos.default(x)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [p_bar_linear_bias, p_bar_linear_weight, x]);  true_graph_0 = false_graph_0 = p_bar_linear_bias = p_bar_linear_weight = x = None
    getitem = conditional[0];  conditional = None
    add = torch.ops.aten.add.Tensor(cos, getitem);  cos = getitem = None
    return (add,)""",
        )

        cond_top_level_nn_module_stack = [
            node.meta["nn_module_stack"]
            for node in ep.graph.nodes
            if node.name == "true_graph_0"
        ][0]

        self.assertTrue(
            "test_cond_with_module_stack_export_with.<locals>.Bar"
            in str(cond_top_level_nn_module_stack)
        )

    # TODO: See https://github.com/pytorch/pytorch/issues/115790
    @unittest.expectedFailure
    def test_cond_with_module_stack_export_with_unflatten(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_fn(x):
                    return self.linear(x).cos()

                def false_fn(x):
                    return self.linear(x).sin()

                return torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])

        class CondExport(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x.cos() + self.bar(x)

        inp = (torch.randn(4, 4),)
        ep = torch.export.export(CondExport(), inp, strict=False)

        cond_top_level_nn_module_stack = [
            node.meta["nn_module_stack"]
            for node in ep.graph.nodes
            if node.name == "true_graph_0"
        ][0]

        # we can't preserve nn_module_stack for the subgraphs for now.
        for node in ep.graph_module.true_graph_0.graph.nodes:
            self.assertEqual(
                node.meta["nn_module_stack"], cond_top_level_nn_module_stack
            )

        # this doesn't work today
        gm_unflat_strict = unflatten(ep)

    def test_predispatch_cond(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("pred", torch.tensor(False))
                self.register_buffer("t", torch.tensor(10))

            def forward(self, x, y):
                def true_fn(x, y):
                    with torch.enable_grad():
                        return x - 1 + self.t + y

                return torch.cond(
                    self.pred,
                    true_fn,
                    lambda x, y: x + 1 - self.t + y,
                    [x, y],
                )

        model = Model()
        with torch.no_grad():
            exported_program = torch.export._trace._export(
                model,
                (torch.tensor(10), torch.tensor(12)),
                {},
                dynamic_shapes=None,
                pre_dispatch=True,
                strict=False,
            )

        self.assertExpectedInline(
            str(exported_program.graph_module.code.strip()),
            """\
def forward(self, b_pred, b_t, x, y):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    conditional = torch.ops.higher_order.cond(b_pred, true_graph_0, false_graph_0, [b_t, x, y]);  b_pred = true_graph_0 = false_graph_0 = b_t = x = y = None
    getitem = conditional[0];  conditional = None
    return (getitem,)""",
        )  # noqa: B950

        self.assertExpectedInline(
            str(exported_program.graph_module.true_graph_0.code.strip()),
            """\
def forward(self, b_t, x, y):
    submod_3 = self.submod_1
    add_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(True, submod_3, x, b_t, y);  submod_3 = x = b_t = y = None
    return (add_1,)""",
        )

        self.assertExpectedInline(
            str(exported_program.graph_module.true_graph_0.submod_1.code.strip()),
            """\
def forward(self, x, b_t, y):
    sub = torch.ops.aten.sub.Tensor(x, 1);  x = None
    add = torch.ops.aten.add.Tensor(sub, b_t);  sub = b_t = None
    add_1 = torch.ops.aten.add.Tensor(add, y);  add = y = None
    return add_1""",
        )

    def test_predispatch_grad_wrappers(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                with torch.enable_grad():
                    x = x - y
                with torch.no_grad():
                    x = x + y
                return x

        # no grad
        model = Model()
        with torch.no_grad():
            ep_nograd = torch.export._trace._export(
                model,
                (torch.tensor(10), torch.tensor(12)),
                {},
                dynamic_shapes=None,
                pre_dispatch=True,
                strict=False,
            )
        # check that only sub op is wrapped with grad_enabled
        getattr_nodes = [
            node for node in ep_nograd.graph.nodes if node.op == "get_attr"
        ]
        self.assertEqual(len(getattr_nodes), 1)
        grad_subgraph = getattr(ep_nograd.graph_module, getattr_nodes[0].target)
        op_node = [
            node for node in grad_subgraph.graph.nodes if node.op == "call_function"
        ][0]
        self.assertEqual(op_node.target._name, "aten::sub.Tensor")

        # enable grad
        model = Model()
        ep_grad = torch.export._trace._export(
            model,
            (torch.tensor(10), torch.tensor(12)),
            {},
            dynamic_shapes=None,
            pre_dispatch=True,
            strict=False,
        )
        # check that only add op is wrapped with grad_enabled
        getattr_nodes = [node for node in ep_grad.graph.nodes if node.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 1)
        grad_subgraph = getattr(ep_grad.graph_module, getattr_nodes[0].target)
        op_node = [
            node for node in grad_subgraph.graph.nodes if node.op == "call_function"
        ][0]
        self.assertEqual(op_node.target._name, "aten::add.Tensor")

    @testing.expectedFailureRetraceability
    def test_layer_sharing(self):
        N, C, H, W = 1, 2, 2, 3

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layer = torch.nn.LayerNorm([C, H, W])
                self.norms = torch.nn.ModuleList(
                    [
                        layer,
                        layer,
                    ]
                )

            def forward(self, x):
                for norm in self.norms:
                    x = norm(x)
                return x

        m = Module()
        copied_m = copy.deepcopy(m)
        ep = export(copied_m, (torch.randn(N, C, H, W),))
        self.assertEqual(copied_m.state_dict(), m.state_dict())
        self.assertEqual(ep.state_dict, m.state_dict())

    @testing.expectedFailureTrainingIRToRunDecomp  # T193692674
    def test_non_persistent_buffer(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("foo", torch.rand(2, 3), persistent=False)

            def forward(self, x):
                return self.foo + x

        inp = torch.rand(2, 3)
        m = MyModule()
        ep = export(m, (inp,), {})

        self.assertEqual(ep.module()(inp), m(inp))
        # Non-persistent buffers should not show up in the state dict
        self.assertNotIn("foo", ep.state_dict)
        named_buffers = {name: buffer for (name, buffer) in ep.named_buffers()}
        # But they should show up in named_buffers()
        self.assertIn("foo", named_buffers)
        self.assertIn("foo", ep.constants)
        self.assertEqual(len(ep.constants), 1)

        # Check the same properties of the unlifted module
        mod = ep.module()
        self.assertNotIn("foo", mod.state_dict())
        mod_named_buffers = {name: buffer for (name, buffer) in mod.named_buffers()}
        self.assertIn("foo", mod_named_buffers)
        self.assertIn("foo", ep.constants)
        self.assertEqual(len(ep.constants), 1)
        self.assertEqual(mod(inp), m(inp))

    def test_export_as_backend(self):
        def f(x, y):
            return x + y

        def my_custom_backend(gm, example_inputs):
            gm = (
                torch.export.export(gm, tuple(example_inputs), strict=False)
                .run_decompositions()
                .module()
            )
            return gm

        inp = (torch.randn(3, 3), torch.randn(3, 3))
        new_res = torch.compile(f, backend=my_custom_backend)(*inp)
        self.assertTrue(torch.allclose(f(*inp), new_res))

    def test_nonstrict_retrace_preserves_metadata(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        inp = torch.randn(4, 4)
        m = MyModule()
        ep = torch.export.export(m, (inp,), {}, strict=False)
        # retrace
        ep2 = torch.export.export(ep.module(), (inp,), {}, strict=False)

        for n1, n2 in zip(list(ep.graph.nodes), list(ep2.graph.nodes)):
            self.assertEqual(n1.meta.get("stack_trace"), n2.meta.get("stack_trace"))

    # TODO Retracing a module with constant attrs don't work.(T193692674)
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_fake_weights(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Parameter(torch.randn(4, 4))
                self.register_buffer("bar", torch.randn(4, 4), persistent=False)
                self.register_buffer("baz", torch.randn(4, 4), persistent=True)

            def forward(self, x):
                return self.foo + x + self.bar + self.baz

        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        with fake_mode:
            m = MyModule()
        inp = torch.randn(4, 4)
        ep = export(m, (inp,))
        # Can't compare outputs because the module has fake weights.

    def test_fake_inputs(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return self.foo + x

        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        m = MyModule()
        with fake_mode:
            inp = torch.randn(4, 4)

        ep = export(m, (inp,))
        self.assertEqual(ep.module()(torch.ones(4, 4)), m(torch.ones(4, 4)))

    def test_trace_under_fake(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return self.foo + x

        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        with fake_mode:
            m = MyModule()
            inp = torch.randn(4, 4)
            # Can't use unqualified export() as it will attempt to deserialize
            # under a new FakeTensorMode.
            ep = torch.export.export(m, (inp,))

    # Errors because non-strict is not supported in training IR (T193692164)
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_compiling_state(self):
        class TestModule1(torch.nn.Module):
            def forward(self, x):
                if torch._dynamo.is_compiling():
                    return x * 2
                else:
                    return x * 3

        class TestModule2(torch.nn.Module):
            def forward(self, x):
                if torch._utils.is_compiling():
                    return x * 2
                else:
                    return x * 3

        class TestModule3(torch.nn.Module):
            def forward(self, x):
                if torch.compiler.is_compiling():
                    return x * 2
                else:
                    return x * 3

        for m in [TestModule1(), TestModule2(), TestModule3()]:
            input = torch.randn(5)
            ep_strict = export(m, (input,), strict=True)
            ep_non_strict = export(m, (input,), strict=False)

            self.assertTrue(torch.allclose(input * 3, m(input)))
            self.assertTrue(torch.allclose(input * 2, ep_strict.module()(input)))
            self.assertTrue(torch.allclose(input * 2, ep_non_strict.module()(input)))

    def test_user_input_and_buffer_mutation(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("foo", torch.randn(4, 4))

            def forward(self, x):
                self.foo.add_(1)
                x.add_(1)
                return self.foo + x

        mod = MyModule()
        mod_copy = copy.deepcopy(mod)
        ep = export(mod_copy, (torch.rand(4, 4),))

        self.assertEqual(mod.foo, ep.module().foo)
        self.assertEqual(mod(torch.ones(4, 4)), ep.module()(torch.ones(4, 4)))

    @testing.expectedFailureTrainingIRToRunDecomp  # T193702033
    def test_symint_tensor_return(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib.returns_tensor_symint(x)[0]

        self._test_export_same_as_eager(Module(), (torch.randn(4, 4),))

    def test_custom_op_auto_functionalize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, z):
                return torch.ops.testlib.foo(x, z)

        inps = (torch.ones(5), torch.ones(5))
        inps_for_export = (torch.ones(5), torch.ones(5))
        inps_for_export_with_decomp = (torch.ones(5), torch.ones(5))

        ep = torch.export.export(M(), inps_for_export)
        x_new_eager, z_new_eager, legit_eager = M()(*inps)
        x_new_export, z_new_export, legit_export = ep.module()(*inps_for_export)
        self.assertTrue(torch.allclose(x_new_eager, x_new_export))
        self.assertTrue(torch.allclose(z_new_eager, z_new_export))
        self.assertTrue(torch.allclose(legit_eager, legit_export))

        ep = ep.run_decompositions()
        x_new_export, z_new_export, legit_export = ep.module()(
            *inps_for_export_with_decomp
        )
        self.assertTrue(torch.allclose(x_new_eager, x_new_export))
        self.assertTrue(torch.allclose(z_new_eager, z_new_export))
        self.assertTrue(torch.allclose(legit_eager, legit_export))

    def test_custom_op_auto_functionalize_pre_dispatch(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.testlib.foo_mutated(x)

        inps = (torch.ones(5),)

        ep = torch.export.export(M(), inps)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    cos = torch.ops.aten.cos.default(x)
    auto_functionalized = torch._higher_order_ops.auto_functionalize.auto_functionalized(torch.ops.testlib.foo.default, x = x, z = cos);  x = cos = None
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    cos_1 = torch.ops.aten.cos.default(getitem_3)
    return (getitem_3, getitem_3, cos_1)""",
        )

        ep = torch.export._trace._export(M(), inps, pre_dispatch=True)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    cos = torch.ops.aten.cos.default(x)
    auto_functionalized = torch._higher_order_ops.auto_functionalize.auto_functionalized(torch.ops.testlib.foo.default, x = x, z = cos);  x = cos = None
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    cos_1 = torch.ops.aten.cos.default(getitem_3)
    return (getitem_3, getitem_3, cos_1)""",
        )

    def test_custom_op_auto_warn_pre_dispatch(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.testlib.foo_functional(x)

        inps = (torch.ones(5),)

        ep = torch.export.export(M(), inps).run_decompositions()
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    cos = torch.ops.aten.cos.default(x)
    cos_1 = torch.ops.aten.cos.default(x);  x = None
    auto_functionalized = torch._higher_order_ops.auto_functionalize.auto_functionalized(torch.ops.testlib.foo.default, x = cos, z = cos_1);  cos = cos_1 = None
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    cos_2 = torch.ops.aten.cos.default(getitem_3);  getitem_3 = None
    return (cos_2,)""",
        )

        ep = torch.export._trace._export(M(), inps, pre_dispatch=True)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    foo_functional = torch.ops.testlib.foo_functional.default(x);  x = None
    return (foo_functional,)""",
        )

    # original input names aren't retraceable:
    # compilation will succeed, but names won't match forward() signature.
    # TODO Retracing a module with constant attrs don't work.(T193692674)
    @testing.expectedFailureRetraceability
    @testing.expectedFailureTrainingIRToRunDecomp
    def test_placeholder_naming_collisions(self):
        # test collisions between nested user inputs
        class Foo(torch.nn.Module):
            def forward(self, x, x_foo, x_foo_0):
                return x["foo"][0] + x_foo[0] + x_foo_0

        inputs = (
            {"foo": [torch.randn(4, 4)]},
            (torch.randn(4, 4),),
            torch.randn(4, 4),
        )
        ep = export(Foo(), inputs)
        expected_names = ["x_foo_0", "x_foo_0_1", "x_foo_0_2"]
        real_names = [spec.arg.name for spec in ep.graph_signature.input_specs]
        self.assertEqual(expected_names, real_names)

        # test collisions between user inputs and params, buffers, constants
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(4))
                self.register_buffer("alpha", torch.randn(4), persistent=True)
                self.register_buffer("beta", torch.randn(4), persistent=False)
                self.gamma = torch.randn(4)

            def forward(self, p, b_alpha, b, c_gamma):
                p = p["param"] + self.param
                b = self.alpha + self.beta + b_alpha + b["beta"]
                c = self.gamma + c_gamma
                return p, b, c

        inputs = (
            {"param": torch.randn(4)},
            torch.randn(4),
            {"beta": torch.randn(4)},
            torch.randn(4),
        )
        ep = export(Foo(), inputs)
        expected_names = [  # user inputs should be prioritized, unprefixed
            ("p_param_1", InputKind.PARAMETER),
            ("b_alpha_1", InputKind.BUFFER),
            ("b_beta_1", InputKind.BUFFER),
            ("c_gamma_1", InputKind.CONSTANT_TENSOR),
            ("p_param", InputKind.USER_INPUT),
            ("b_alpha", InputKind.USER_INPUT),
            ("b_beta", InputKind.USER_INPUT),
            ("c_gamma", InputKind.USER_INPUT),
        ]
        real_names = [
            (spec.arg.name, spec.kind) for spec in ep.graph_signature.input_specs
        ]
        self.assertEqual(expected_names, real_names)

        # test collisions between user inputs & call_function nodes
        class Foo(torch.nn.Module):
            def forward(self, mul, add, add_1):
                return mul * mul + add * add_1

        ep = export(Foo(), (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)))
        expected_names_and_ops = [
            ("mul", "placeholder"),
            ("add", "placeholder"),
            ("add_1", "placeholder"),
            ("mul_1", "call_function"),
            ("mul_2", "call_function"),
            ("add_2", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [(node.name, node.op) for node in ep.graph.nodes]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)

    @testing.expectedFailureRetraceability
    def test_placeholder_naming_collisions_hoo_subgraphs(self):
        # test collisions between user inputs, top-level nodes, and HOO subgraph nodes
        class Foo(torch.nn.Module):
            def forward(self, x, mul, mul_1):
                _mul = x * x
                y = cond(
                    _mul.sum() > 0,
                    lambda x, y, z: x * y * z,
                    lambda x, y, z: x + y + z,
                    [_mul, mul, mul_1],
                )
                with torch.enable_grad():
                    y = y * y
                return y

        with torch.no_grad():
            ep = torch.export._trace._export(
                Foo(),
                (torch.randn(4), torch.randn(4), torch.randn(4)),
                pre_dispatch=True,
            )
        # test cond subgraph
        expected_names_and_ops = [
            ("mul_2", "placeholder"),
            ("mul", "placeholder"),
            ("mul_1", "placeholder"),
            ("mul_3", "call_function"),
            ("mul_4", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [
            (node.name, node.op) for node in ep.graph_module.true_graph_0.graph.nodes
        ]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)
        # test set_grad_enabled subgraph
        expected_names_and_ops = [
            ("getitem", "placeholder"),
            ("mul_1", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [
            (node.name, node.op) for node in ep.graph_module.submod_1.graph.nodes
        ]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)

        # test collisions between user inputs & higher order op subgraphs
        # (please never do this)
        class Foo(torch.nn.Module):
            def forward(self, input, true_graph, body_graph):
                def map_body(x, y):
                    return x + y

                x = map(map_body, input, body_graph[0])
                x = x + true_graph[0] + true_graph[1]
                x = cond(x.sum() > 0, lambda x: x * 2.0, lambda x: x + 2.0, [x])
                x = cond(x.sum() > 0, lambda x: x * 2.0, lambda x: x + 2.0, [x])
                return x

        inputs = (
            torch.randn(10, 4),
            (torch.randn(4), torch.randn(4)),
            (torch.randn(4),),
        )
        ep = export(Foo(), inputs)
        expected_getattr_names = [
            "body_graph_1",
            "true_graph_2",
            "false_graph_0",
            "true_graph_3",
            "false_graph_1",
        ]
        real_getattr_names = [
            node.name for node in ep.graph.nodes if node.op == "get_attr"
        ]
        self.assertEqual(expected_getattr_names, real_getattr_names)

    def test_constant_input_naming(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, div="floor"):
                return torch.div(x, y, rounding_mode=div)

        f = Foo()
        inputs = (torch.randn(4), torch.randn(4), "floor")
        ep = export(f, inputs)
        div_spec = ep.graph_signature.input_specs[2]
        self.assertEqual(div_spec.arg.name, "div")
        self.assertEqual(div_spec.arg.value, "floor")

    def test_unbacked_deferred_runtime_retrace(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                y_sum = y.sin().sum()
                with torch.no_grad():
                    a = x.item()
                    torch._check_is_size(a)
                    torch._check(a > 2)
                    torch._check(a < 6)
                    unbacked_shape = torch.ops.testlib.foo_unbacked(a)
                return y + y_sum + unbacked_shape.sum()

        inps = (torch.tensor(4), torch.randn(5, 5))
        from torch.export import _trace

        ep_pre = _trace._export(Foo(), inps, pre_dispatch=True, strict=False)
        self.assertExpectedInline(
            str(ep_pre.graph_module.submod_1.code).strip(),
            """\
def forward(self, x):
    item = torch.ops.aten.item.default(x);  x = None
    sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(item)
    sym_constrain_range_default = torch.ops.aten.sym_constrain_range.default(item, min = 3, max = 5)
    ge = item >= 0
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression 0 <= u1 on node 'ge'");  ge = None
    gt = item > 2
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression 2 < u1 on node 'gt'");  gt = None
    lt = item < 6
    _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(lt, "Runtime assertion failed for expression u1 < 6 on node 'lt'");  lt = None
    foo_unbacked = torch.ops.testlib.foo_unbacked.default(item);  item = None
    return foo_unbacked""",
        )
        ep_aot = ep_pre.run_decompositions()
        self.assertExpectedInline(
            str(ep_aot.graph_module.code).strip(),
            """\
def forward(self, x, y):
    sin = torch.ops.aten.sin.default(y)
    sum_1 = torch.ops.aten.sum.dim_IntList(sin, []);  sin = None
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x);  x = None
    sym_constrain_range_for_size = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense)
    sym_constrain_range = torch.ops.aten.sym_constrain_range.default(_local_scalar_dense, min = 3, max = 5)
    ge = _local_scalar_dense >= 0
    _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression 0 <= u1 on node 'ge'");  ge = None
    gt = _local_scalar_dense > 2
    _assert_scalar_1 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression 2 < u1 on node 'gt'");  gt = None
    lt = _local_scalar_dense < 6;  _local_scalar_dense = None
    _assert_scalar_2 = torch.ops.aten._assert_scalar.default(lt, "Runtime assertion failed for expression u1 < 6 on node 'lt'");  lt = None
    full = torch.ops.aten.full.default([4, 4], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(y, sum_1);  y = sum_1 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(full, []);  full = None
    add_1 = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
    return (add_1,)""",
        )

    def test_nested_dynamic_shapes_spec(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                (a0, a1), (b0, b1), (c0, c1, c2) = x
                return a0 + a1 + b0 + b1 + c0 + c1 + c2

        f = Foo()
        inputs = (
            (1, 2),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
        )
        # make sure this gets parsed correctly as 7 individual inputs, not 3 tensors
        dynamic_shapes = {
            "x": (
                (None, None),
                (None, None),
                (None, None, None),
            )
        }
        export(f, (inputs,), dynamic_shapes=dynamic_shapes)

    def test_disable_forced_specializations_ok(self):
        # check that _disable_forced_specializations and _allow_complex_guards_as_runtime_asserts flags
        # both behave correctly, avoiding forced specializations and deferring to runtime.
        # case 1: modulo guards
        from torch.export import dims

        class Mod4Reshape(torch.nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[0] - 1, 4, -1)  # Mod(s0*s1, 4*(s0-1)) = 0

        inputs = (torch.randn(10, 72),)
        dx, dy = dims("dx", "dy")
        with self.assertRaisesRegex(  # this will force specialize
            torch._dynamo.exc.UserError,
            r".*Specializations unexpectedly required(.*\n)*"
            r".*dx = .* must be specialized to 10 because the guards generated for it are too complex(.*\n)*"
            r".*dy = .* must be specialized to 72 because the guards generated for it are too complex(.*\n)*",
        ):
            export(
                Mod4Reshape(),
                inputs,
                dynamic_shapes={"x": (dx, dy)},
            )

        torch.export._trace._export(  # just check this successfully compiles
            Mod4Reshape(),
            inputs,
            dynamic_shapes={"x": (dx, dy)},
            strict=False,
            _disable_forced_specializations=True,
        )
        ep = torch.export._trace._export(
            Mod4Reshape(),
            inputs,
            dynamic_shapes={"x": (dx, dy)},
            _allow_complex_guards_as_runtime_asserts=True,
        )
        out1 = ep.module()(torch.randn(8, 7))
        self.assertEqual(out1.shape, torch.ones(7, 4, 2).shape)
        out2 = ep.module()(torch.randn(12, 11))
        self.assertEqual(out2.shape, torch.ones(11, 4, 3).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(Mod\(s0\*s1, 4\*s0 \- 4\), 0\) on node 'eq.*'",
        ):
            ep.module()(torch.randn(8, 8))  # fail

        # case 2: 2d reshape
        class FreeReshape(torch.nn.Module):
            def forward(self, x, y, z):
                return x.reshape([-1]) + y.reshape([-1]) + z  # s0*s1 = s2*s3 = s4

        inputs = (
            torch.randn(6, 8),
            torch.randn(3, 16),
            torch.randn(48),
        )
        dynamic_shapes = {
            "x": [Dim(f"dx{i}", min=2) for i in range(2)],
            "y": [Dim(f"dy{i}", min=2) for i in range(2)],
            "z": [Dim(f"dz{i}", min=4) for i in range(1)],
        }
        with self.assertRaisesRegex(  # this will force specialize
            torch._dynamo.exc.UserError,
            r".*Specializations unexpectedly required(.*\n)*"
            r".*dx0 = .* must be specialized to 6 because the guards generated for it are too complex(.*\n)*"
            r".*dx1 = .* must be specialized to 8 because the guards generated for it are too complex(.*\n)*",
        ):
            export(
                FreeReshape(),
                inputs,
                dynamic_shapes=dynamic_shapes,
            )
        torch.export._trace._export(
            FreeReshape(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            _disable_forced_specializations=True,
        )
        ep = torch.export._trace._export(
            FreeReshape(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            _allow_complex_guards_as_runtime_asserts=True,
        )
        out1 = ep.module()(torch.randn(48, 1), torch.randn(4, 12), torch.randn(48))
        self.assertEqual(out1.shape, torch.ones(48).shape)
        out2 = ep.module()(torch.randn(5, 8), torch.randn(4, 10), torch.randn(40))
        self.assertEqual(out2.shape, torch.ones(40).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s0\*s1, s2\*s3\) on node 'eq.*'",
        ):  # fail only at runtime
            ep.module()(torch.randn(5, 8), torch.randn(4, 5), torch.randn(30))  # fail

        # case 3: 3d reshape (previously failing with different issue)
        class Reshape3d(torch.nn.Module):
            def forward(self, x, y):
                return x.reshape([-1]) + y  # s0*s1*s2 = s3

        inputs = (
            torch.randn(4, 3, 2),
            torch.randn(24),
        )
        dynamic_shapes = {
            "x": (Dim("dx0", min=2), Dim("dx1", min=2), Dim("dx2", min=2)),
            "y": (Dim("dy", min=8),),
        }
        with self.assertRaisesRegex(  # this will force specialize
            torch._dynamo.exc.UserError,
            r".*Specializations unexpectedly required(.*\n)*"
            r"Suggested fixes:(.*\n)*"
            r".*dx0 = 4(.*\n)*"
            r".*dx1 = 3(.*\n)*"
            r".*dx2 = 2(.*\n)*"
            r".*dy = 24(.*\n)*",
        ):
            export(
                Reshape3d(),
                inputs,
                dynamic_shapes=dynamic_shapes,
            )

        torch.export._trace._export(
            Reshape3d(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            _disable_forced_specializations=True,
        )
        ep = torch.export._trace._export(
            Reshape3d(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            _allow_complex_guards_as_runtime_asserts=True,
        )
        out1 = ep.module()(torch.randn(9, 7, 2), torch.randn(126))
        self.assertEqual(out1.shape, torch.ones(126).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s0\*s1\*s2, s3\) on node 'eq.*'",
        ):  # fail only at runtime
            ep.module()(torch.randn(4, 3, 2), torch.randn(10))  # fail

    def test_disable_forced_specializations_errors(self):
        # check error messages with disable_forced_specializations = False/True
        class Foo(torch.nn.Module):
            def forward(self, w, x, y, z):
                return w.reshape([-1]) + x, y + z  # simple: s0*s1 = s2, s3 = s4

        inputs = (
            torch.randn(3, 4),
            torch.randn(12),
            torch.randn(4),
            torch.randn(4),
        )
        dynamic_shapes = {
            "w": [Dim(f"dw{i}") for i in range(2)],
            "x": [Dim(f"dx{i}") for i in range(1)],
            "y": [Dim("dy")],  # y & z incorrect, export is supposed to fail.
            "z": [Dim("dz")],  # suggested fix should be to match these up.
        }
        with self.assertRaisesRegex(  # if allow = False, suggested fixes should specialize 3, 4, 12.
            torch._dynamo.exc.UserError,
            r".*Specializations unexpectedly required(.*\n)*"
            r"Suggested fixes:(.*\n)*"
            r".*dw0 = 3(.*\n)*"
            r".*dw1 = 4(.*\n)*"
            r".*dx0 = 12(.*\n)*"
            r".*dz = dy(.*\n)*",
        ):
            torch.export._trace._export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                _disable_forced_specializations=False,
            )
        with self.assertRaisesRegex(  # if disable=True, suggested fixes should not specialize.
            torch._dynamo.exc.UserError,
            r".*Constraints violated(.*\n)*"
            r"Suggested fixes:(.*\n)*"
            r".*dz = dy(.*\n)*",
        ) as msg:
            torch.export._trace._export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                _disable_forced_specializations=True,
            )

    # TODO requires_grad doesn't seem to work with serialization.
    @testing.expectedFailureSerDer
    def test_preserve_requires_grad_placeholders(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, x, y):
                return self.p + x + y

        m = Module()
        ep = export(m, (torch.randn(3, 3), torch.randn(3, 3, requires_grad=True)))
        placeholders = [
            node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
        ]
        self.assertTrue(placeholders[0].meta["val"].requires_grad)
        self.assertFalse(placeholders[1].meta["val"].requires_grad)
        self.assertTrue(placeholders[2].meta["val"].requires_grad)

    def test_reshape_view_helper(self):
        # see: https://github.com/pytorch/pytorch/issues/126607
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x.view(x.size(1), -1)
                # torch/_refs/__init__/_reshape_view_helper() will generate guards on reshape kernel(?)
                # Ne(s0, 20), so that reshape isn't no-op
                # Ne(Mod(s0, 20), 0), so that reshape needs to first flatten [s0, 20, 16] -> [s0*20, 16]
                # then split_dim -> [20, s0, 16]
                # check that these show up in graph
                return torch.nn.functional.softmax(
                    x, dim=0
                )  # don't think softmax actually creates any issues, just part of original test

        model = Model()
        x = torch.rand(1024, 20, 16)
        dynamic_shapes = {"x": {0: Dim("batch")}}
        ep = torch.export._trace._export(
            model,
            (x,),
            dynamic_shapes=dynamic_shapes,
            _allow_complex_guards_as_runtime_asserts=True,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, 20\)",
        ):
            ep.module()(torch.randn(20, 20, 16))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(Mod\(s0, 20\), 0\)",
        ):
            ep.module()(torch.randn(400, 20, 16))
        ep.module()(torch.randn(42, 20, 16))

    def test_allow_explicit_guards_as_runtime_asserts(self):
        # check that explicit guards are treated as runtime assertions
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                # check that negation of first guard also shows up as runtime assertion
                if x.shape[0] == y.shape[0]:  # False
                    return x + y
                elif x.shape[0] == y.shape[0] ** 3:  # False
                    return x + 2, y + 3
                elif x.shape[0] ** 2 == y.shape[0] * 3:  # True
                    return x * 2.0, y * 3.0

        inputs = (torch.randn(6), torch.randn(12))
        dynamic_shapes = {"x": [Dim("dx", min=4)], "y": [Dim("dy", min=4)]}
        ep = torch.export._trace._export(
            Foo(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            _allow_complex_guards_as_runtime_asserts=True,
        )
        # check forward pass
        out0, out1 = ep.module()(torch.randn(9), torch.randn(27))
        self.assertEqual(out0.shape, torch.ones(9).shape)
        self.assertEqual(out1.shape, torch.ones(27).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, s1\)",
        ):  # fail only at runtime
            ep.module()(torch.randn(4), torch.randn(4))  # fail
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, s1\**3\)",
        ):
            ep.module()(torch.randn(64), torch.randn(4))  # fail
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s0\**2, 3\*s1\)",
        ):
            ep.module()(torch.randn(10), torch.randn(9))  # fail

        # this should be set with command line flag TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1,
        # but dynamo checks that at torch import time, so setting os.environ makes no difference
        # instead, manually patch dynamo config and test.
        # test that setting this flag removes runtime asserts
        from torch._dynamo import config as _dynamo_config

        with _dynamo_config.patch(
            do_not_emit_runtime_asserts=True,
        ):
            ep = torch.export._trace._export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                _allow_complex_guards_as_runtime_asserts=True,
            ).run_decompositions()

        self.assertEqual(
            [
                node.target == torch.ops.aten._assert_scalar.default
                for node in ep.graph.nodes
            ].count(True),
            0,
        )

    def test_constant_aliasing(self):
        class M1(torch.nn.Module):
            def __init__(self, m2, foo):
                super().__init__()
                self.m2 = m2
                self.foo = foo

            def forward(self, x):
                return x + self.foo + self.m2(x)

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.ones(3, 3)

            def forward(self, x):
                return x + self.foo

        m2 = M2()
        m1 = M1(m2, m2.foo)
        inps = (torch.ones(3, 3),)
        ep = torch.export.export(m1, inps, strict=False)
        # check both constants appear in list
        self.assertEqual(sorted(list(ep.constants)), ["foo", "m2.foo"])
        # check only one input spec exists
        num_constant_inputs = [
            spec.kind == InputKind.CONSTANT_TENSOR
            for spec in ep.graph_signature.input_specs
        ].count(True)
        self.assertEqual(num_constant_inputs, 1)
        # unflatten
        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(m1(*inps), unflattened(*inps)))

    @testing.expectedFailureRetraceability
    def test_unused_aliases(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # param
                self.alpha = torch.nn.Parameter(torch.randn(4))
                self.beta = self.alpha
                self.gamma = self.alpha

            def forward(self, x):
                return x + self.gamma

        inps = (torch.randn(4),)
        ep = export(Foo(), inps)
        # placeholder nodes will be deduplicated in strict-mode,
        # but check that all params still appear in state dict
        for param in ["alpha", "beta", "gamma"]:
            self.assertTrue(param in ep.state_dict)

        # check that they also appear in unflattened state dict
        unep = unflatten(ep)
        for param in ["alpha", "beta", "gamma"]:
            self.assertTrue(param in unep.state_dict())


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestOneOffModelExportResult(TestCase):
    def test_scaled_dot_product_attention_cpu(self):
        """
        This test makes sure we are always getting the same decomposition result for SDPA.
        As of now _scaled_dot_product_flash_attention_for_cpu is expected to show up in
        export() result. Some downstream backend then further decompose it into core ATen
        ops in torch/_decomp/decompositions.py (search for
        _scaled_dot_product_flash_attention_for_cpu).

        Export is decomposing based on the CompositeImplicitAutograd kernel implementation
        of SDPA. If this test fails, it means the kernel is being modified. In this case
        we strongly encourage you to change the decomposition rule under
        torch/_decomp/decompositions.py along with the kernel changes, so all of the
        downstream backends are not being affected.
        """

        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, None, dropout_p=0.0, is_causal=True
                )
                return attn_output

        q = torch.randn(1, 1, 8, 8, device="cpu")
        k = torch.randn(1, 1, 8, 8, device="cpu")
        v = torch.randn(1, 1, 8, 8, device="cpu")

        from torch.nn.attention import SDPBackend

        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
            ep = torch.export.export(ScaledDotProductAttention(), (q, k, v))
            print(ep.graph)
            ep.run_decompositions()
            print(ep.graph)

    #         self.assertExpectedInline(ep.graph_module.code.strip(), """\
    # def forward(self, arg0_1, arg1_1, arg2_1):
    #     _scaled_dot_product_flash_attention_for_cpu = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(arg0_1, arg1_1, arg2_1, 0.0, True);  arg0_1 = arg1_1 = arg2_1 = None
    #     getitem = _scaled_dot_product_flash_attention_for_cpu[0];  _scaled_dot_product_flash_attention_for_cpu = None
    #     return (getitem,)""")

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_scaled_dot_product_attention_cuda(self):
        """
        This test makes sure we are always getting the same decomposition result for SDPA.
        As of now _scaled_dot_product_flash_attention is expected to show up in
        export() result (GPU tensors are given). Currently there's no downstream
        backend relies on this export result so if this test fails, feel free to
        change it to the latest export() result.
        """

        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, None, dropout_p=0.0, is_causal=True
                )
                return attn_output

        q = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")

        ep = torch.export.export(
            ScaledDotProductAttention(), (q, k, v)
        ).run_decompositions()
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, q, k, v):
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True, scale = 0.125);  q = k = v = None
    getitem = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
    return (getitem,)""",
        )

    def test_int_list_output(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return [((1, 3), [x + x, x * x])]

        ep = torch.export.export(M(), (torch.ones(2, 3),))
        res = ep.module()(torch.ones(2, 3))
        self.assertEqual(res[0][0], (1, 3))

    def test_primitive_constant_output(self):
        class Z(torch.nn.Module):
            def forward(self, x, y):
                return y * x

        ep = torch.export.export(Z(), (torch.tensor(3), 5))
        res = ep.module()(torch.tensor(4), 5)
        self.assertEqual(res, torch.tensor(20))

        class B(torch.nn.Module):
            def forward(self, x, y):
                return y * x, y

        ep = torch.export.export(B(), (torch.tensor(3), 5))
        res = ep.module()(torch.tensor(4), 5)
        self.assertEqual(res[0], torch.tensor(20))
        self.assertEqual(res[1], 5)

        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5, but got 20"),
        ):
            res = ep.module()(torch.tensor(4), 20)

        class F(torch.nn.Module):
            def forward(self, x):
                # return a constant of primitive type
                y = 5
                return y * x, y

        ep = torch.export.export(F(), (torch.tensor(3),))
        res = ep.module()(torch.tensor(4))
        self.assertEqual(res[0], torch.tensor(20))
        self.assertEqual(res[1], 5)

        class Q(torch.nn.Module):
            def forward(self, x, y):
                return y * x, y - 1

        ep = torch.export.export(Q(), (torch.tensor(3), 5))
        res = ep.module()(torch.tensor(4), 5)
        self.assertEqual(res[0], torch.tensor(20))
        self.assertEqual(res[1], 4)

    def test_unbacked_sdpa(self):
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from torch.nn.functional import scaled_dot_product_attention

        class Module(torch.nn.Module):
            def forward(
                self, query: torch.Tensor, cache: torch.Tensor, start_pos: torch.Tensor
            ) -> torch.Tensor:
                # x.sizes(): 1, 128, 16, 128
                sp = start_pos.item()
                torch._check_is_size(sp)
                torch._check(sp >= 0)
                torch._check(sp <= 126)
                key = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
                value = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
                query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp#L732
                return scaled_dot_product_attention(query, key, value)

        cache = torch.randn(1, 128, 16, 128, dtype=torch.float16)
        query = torch.randn(1, 1, 16, 128, dtype=torch.float16)
        start_pos = torch.tensor([0])
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            ep = torch.export.export(Module(), (query, cache, start_pos))
            args = (query, cache, start_pos)
            self.assertEqual(ep.module()(*args), Module()(*args))
            args = (query, cache, torch.tensor([3]))
            self.assertEqual(ep.module()(*args), Module()(*args))
            args = (query, cache, torch.tensor([126]))
            self.assertEqual(ep.module()(*args), Module()(*args))

    def test_none_input_output(self):
        class Z(torch.nn.Module):
            def forward(self, x, y):
                return x * x

        ep = torch.export.export(Z(), (torch.tensor(3), None))
        res = ep.module()(torch.tensor(4), None)
        self.assertEqual(res, torch.tensor(16))

        class B(torch.nn.Module):
            def forward(self, x, y):
                return x * x, y

        ep = torch.export.export(B(), (torch.tensor(3), None))
        res = ep.module()(torch.tensor(4), None)
        self.assertEqual(res[0], torch.tensor(16))
        self.assertEqual(res[1], None)

        decomp = ep.run_decompositions()
        gm = decomp.module()
        res = gm(torch.tensor(4), None)
        self.assertEqual(res[0], torch.tensor(16))
        self.assertEqual(res[1], None)

    def test_print(self):
        class M(torch.nn.Module):
            def forward(self, x):
                print("start")
                x1 = x + x
                print(x1)
                x2 = x1 * x1
                print(1, 2, 3)
                x3 = x2 + x2
                return (x1, x3)

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    mul = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
    return (add, add_1)""",
        )

    def test_logging_logger(self):
        logger = logging.getLogger(__name__)

        class M(torch.nn.Module):
            def forward(self, x):
                logger.log("start")
                x1 = x + x
                logger.debug(x1)
                x2 = x1 * x1
                logger.info(1, 2, 3)
                x3 = x2 + x2
                return (x1, x3)

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    mul = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
    return (add, add_1)""",
        )

    @unittest.skipIf(not TEST_TRANSFORMERS, "No transformers")
    def test_hf_logging_logger(self):
        import transformers

        logger = transformers.utils.logging.get_logger(__name__)

        class M(torch.nn.Module):
            def forward(self, x):
                logger.warning_once("start")
                x1 = x + x
                x2 = x1 * x1
                x3 = x2 + x2
                return (x1, x3)

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    mul = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
    return (add, add_1)""",
        )

    def test_warning(self):
        class M(torch.nn.Module):
            def forward(self, x):
                warnings.warn("moo")
                res = x + x
                warnings.warn(f"{res}")
                return res

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    return (add,)""",
        )

    def test_constant_fqn(self):
        class Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = torch.rand(2, 3)
                self.parameter = torch.nn.Parameter(torch.rand(2, 3))

            def forward(self, x):
                return x + self.constant

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = Nested()

            def forward(self, x):
                return self.nested(x) + self.nested.constant + self.nested.parameter

        m = Mod()
        ep = export(m, (torch.rand(2, 3),), strict=True)
        self.assertEqual(ep.constants["nested.constant"], m.nested.constant)
        self.assertEqual(ep.module()(torch.ones(2, 3)), m(torch.ones(2, 3)))

    def test_constant_name(self):
        class Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = torch.rand(2, 3)
                self.parameter = torch.nn.Parameter(torch.rand(2, 3))

            def forward(self, x):
                return x + self.constant

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested_1 = Nested()
                self.nested_2 = Nested()

            def forward(self, x):
                return (
                    self.nested_1(x)
                    + self.nested_2(x)
                    + self.nested_1.constant
                    + self.nested_2.constant
                    + self.nested_1.parameter
                    + self.nested_2.parameter
                )

        m = Mod()
        ep = export(m, (torch.rand(2, 3),), strict=False)
        self.assertEqual(ep.module()(torch.ones(2, 3)), m(torch.ones(2, 3)))

        # check constant fqn when there are multiple instances of the same class
        self.assertEqual(ep.constants["nested_1.constant"], m.nested_1.constant)
        self.assertEqual(ep.constants["nested_2.constant"], m.nested_2.constant)

        # check constant_name in the graph
        placeholders = [
            node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(len(placeholders), 5)
        self.assertTrue(all(ph.name == ph.target for ph in placeholders))
        # suffix should be added to duplicated constant_name
        self.assertEqual(placeholders[2].name, "c_nested_1_constant")
        self.assertEqual(placeholders[3].name, "c_nested_2_constant")

    def test_nested_retrace(self):
        class Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(3))

            def forward(self, x):
                return x + self.param

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = Nested()

            def forward(self, x):
                return x + self.nested(x)

        # first export
        foo = Foo().to("meta")
        inputs = (torch.ones(3, device="meta"),)
        foo(*inputs)
        ep = torch.export.export(foo, inputs, strict=False)

        # second export
        foo_1 = ep.module()
        ep_1 = torch.export.export(foo_1, inputs, strict=False)

        for node1, node2 in zip(ep.graph.nodes, ep_1.graph.nodes):
            nn_module_stack_1 = node1.meta.get("nn_module_stack", None)
            nn_module_stack_2 = node2.meta.get("nn_module_stack", None)

            if nn_module_stack_1 is None:
                self.assertTrue(nn_module_stack_2 is None)
            else:
                for v1, v2 in zip(
                    nn_module_stack_1.values(), nn_module_stack_2.values()
                ):
                    self.assertEqual(v1, v2)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestExportCustomClass(TorchTestCase):
    def setUp(self):
        if IS_FBCODE:
            lib_file_path = "//caffe2/test/cpp/jit:test_custom_class_registrations"
        elif IS_SANDCASTLE or IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
        torch.ops.load_library(str(lib_file_path))

    def test_lift_custom_obj(self):
        # TODO: fix this test once custom class tracing is implemented

        custom_obj = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export(f, inputs)

        # Replace one of the values with an instance of our custom class
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                with ep.graph.inserting_before(node):
                    setattr(ep.graph_module, "custom_obj", custom_obj)
                    getattr_node = ep.graph.get_attr("custom_obj")
                    # Copy over an nn_module_stack as they are required.
                    getattr_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
                    custom_node = ep.graph.call_function(
                        torch.ops._TorchScriptTesting.take_an_instance.default,
                        (getattr_node,),
                    )
                    custom_node.meta["val"] = torch.ones(4, 4)
                    # Copy over an nn_module_stack as they are required.
                    custom_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
                    custom_node.meta["torch_fn"] = (
                        "custom_op",
                        "torch.ops._TorchScriptTesting.take_an_instance.default",
                    )
                    arg0, _ = node.args
                    node.args = (arg0, custom_node)

        from torch._export.passes.lift_constants_pass import lift_constants_pass
        from torch._export.serde.serialize import deserialize, serialize

        constants = lift_constants_pass(ep.graph_module, ep.graph_signature, {})
        for k, v in constants.items():
            assert k not in ep.constants
            ep._constants[k] = v
        serialized_vals = serialize(ep)
        deserialized_ep = deserialize(serialized_vals)

        for node in deserialized_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target
                == torch.ops._TorchScriptTesting.take_an_instance.default
            ):
                arg = node.args[0]
                self.assertTrue(arg.op == "placeholder")

    def test_tolist_nonstrict_output(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x.tolist()

        ep = torch.export.export(M(), (torch.ones(3),), strict=False)


if __name__ == "__main__":
    run_tests()
