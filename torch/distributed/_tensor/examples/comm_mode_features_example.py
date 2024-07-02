import os

from typing import Callable, Dict

import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.examples.comm_mode_features_example_argparser import args

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    MLPModule,
    MLPStacked,
    ModelArgs,
    NUM_DEVICES,
    Transformer,
)


def get_device_type() -> str:
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )


c10d_functional = torch.ops.c10d_functional

aten = torch.ops.aten
supported_ops = [aten.view.default, aten._to_copy.default]


class CommDebugModeExample:
    """
    Checks if the set of keys in ground truth dictionary and the set
    produced in advanced_module_tracker are in the same order
    """

    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        self.device_type = get_device_type()

    def test_MLP_distributed_sharding_display(self) -> None:
        """
        Example of obtaining all module's FQN and parameters for a given distributed model and printing the sharding info

        Expected output:
        MLPModule.net1.weight: (Shard(dim=0),)
        MLPModule.net1.bias: (Shard(dim=0),)
        MLPModule.net2.weight: (Shard(dim=1),)
        MLPModule.net2.bias: (Replicate(),)
        """
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        inp_size = [8, 10]
        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        LR = 0.25

        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        comm_mode.print_sharding_info()

    def test_MLPStacked_distributed_sharding_display(self) -> None:
        """
        Example of obtaining all module's FQN and parameters for a given
        distributed model with nested modules and printing the sharding info

        Expected output:
        MLPStacked.layers.0.net1.weight: (Shard(dim=0),)
        MLPStacked.layers.0.net1.bias: (Shard(dim=0),)
        MLPStacked.layers.0.net2.weight: (Shard(dim=1),)
        MLPStacked.layers.0.net2.bias: (Replicate(),)
        MLPStacked.layers.1.net1.weight: (Shard(dim=0),)
        MLPStacked.layers.1.net1.bias: (Shard(dim=0),)
        MLPStacked.layers.1.net2.weight: (Shard(dim=1),)
        MLPStacked.layers.1.net2.bias: (Replicate(),)
        """
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        inp_size = [8, 10]
        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPStacked(self.device_type)

        LR = 0.25

        parallelize_plan = {
            "MLPStacked.layers.0.net1": ColwiseParallel(),
            "MLPStacked.layers.0.net2": RowwiseParallel(),
            "MLPStacked.layers.1.net1": ColwiseParallel(),
            "MLPStacked.layers.1.net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        comm_mode.print_sharding_info()

    def test_MLP_module_tracing(self) -> None:
        """
        Example code to demonstrate CommModeDebug's module level tracing using a MLP model.
        Prints a table of module level collective tracing information and logs table to output.txt

        Expected Output
        Global
        *c10d_functional.all_reduce: 1
        MLPModule
            *c10d_functional.all_reduce: 1
            MLPModule.net1
            MLPModule.relu
            MLPModule.net2
            *c10d_functional.all_reduce: 1
        """

        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        inp_size = [8, 10]
        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        LR = 0.25

        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        # print the module level collective tracing information
        print(comm_mode.generate_module_tracing_table())
        comm_mode.log_module_tracing_table_to_file()

    def test_transformer_module_tracing(self, is_seq_parallel: bool = False) -> None:
        """
        Example code to demonstrate CommModeDebug's module level tracing using a distributed Transformer model.
        Prints a table of module level collective tracing information and logs table to output.txt

        Expected output:
        Global
        *c10d_functional.all_reduce: 6
        *c10d_functional.all_gather_into_tensor: 1
        Transformer
            *c10d_functional.all_reduce: 6
            *c10d_functional.all_gather_into_tensor: 1
            Transformer.tok_embeddings
            *c10d_functional.all_reduce: 1
            Transformer.pos_embeddings
            *c10d_functional.all_reduce: 1
            Transformer.dropout
            Transformer.layers.0
            *c10d_functional.all_reduce: 2
            Transformer.layers.0.attention_norm
            Transformer.layers.0.attention
                *c10d_functional.all_reduce: 1
                Transformer.layers.0.attention.wq
                Transformer.layers.0.attention.wk
                Transformer.layers.0.attention.wv
                Transformer.layers.0.attention.wo
                *c10d_functional.all_reduce: 1
                Transformer.layers.0.attention.resid_dropout
            Transformer.layers.0.ffn_norm
            Transformer.layers.0.feed_forward
                *c10d_functional.all_reduce: 1
                Transformer.layers.0.feed_forward.w1
                Transformer.layers.0.feed_forward.gelu
                Transformer.layers.0.feed_forward.w2
                *c10d_functional.all_reduce: 1
                Transformer.layers.0.feed_forward.resid_dropout
            Transformer.layers.1
            *c10d_functional.all_reduce: 2
            Transformer.layers.1.attention_norm
            Transformer.layers.1.attention
                *c10d_functional.all_reduce: 1
                Transformer.layers.1.attention.wq
                Transformer.layers.1.attention.wk
                Transformer.layers.1.attention.wv
                Transformer.layers.1.attention.wo
                *c10d_functional.all_reduce: 1
                Transformer.layers.1.attention.resid_dropout
            Transformer.layers.1.ffn_norm
            Transformer.layers.1.feed_forward
                *c10d_functional.all_reduce: 1
                Transformer.layers.1.feed_forward.w1
                Transformer.layers.1.feed_forward.gelu
                Transformer.layers.1.feed_forward.w2
                *c10d_functional.all_reduce: 1
                Transformer.layers.1.feed_forward.resid_dropout
            Transformer.norm
            Transformer.output
            *c10d_functional.all_gather_into_tensor: 1

        """
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        model_args = ModelArgs()
        model = Transformer(model_args).to(device=self.device_type)
        model = Transformer.parallelize(model, device_mesh, is_seq_parallel)

        LR = 0.25
        inp_size = [8, 8]

        torch.manual_seed(0)
        inp = torch.randint(model_args.vocab_size, inp_size, device=self.device_type)

        comm_mode = CommDebugMode()
        with comm_mode:
            output = model(inp)

        # print the module level collective tracing information
        print(comm_mode.generate_module_tracing_table())
        comm_mode.log_module_tracing_table_to_file()

    def test_MLP_operation_tracing(self) -> None:
        """
        Example code to demonstrate CommModeDebug's module operation level tracing using a distributed MLP model.
        Prints a table of module opoeration level collective tracing information and logs table to output.txt

        Expected output:
        Global
          *c10d_functional.all_reduce: 1
          MLPModule
            *c10d_functional.all_reduce: 1
            MLPModule.net1
              FORWARD PASS
                *aten.detach.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([16])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([16])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([16])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                *aten.view.default
                *aten.t.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.t.default
                *aten.addmm.default
                  shape: [torch.Size([16]), torch.Size([8, 10]), torch.Size([10, 16])]
                  sharding: [(Shard(dim=0),), (Replicate(),), (Shard(dim=1),)]
                *aten.addmm.default
                *aten.view.default
            MLPModule.relu
              FORWARD PASS
                *aten.relu.default
                *aten.detach.default
            MLPModule.net2
              *c10d_functional.all_reduce: 1
              FORWARD PASS
                *aten.detach.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([10])]
                  sharding: [(Replicate(),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([10])]
                  sharding: [(Replicate(),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([10])]
                  sharding: [(Replicate(),)]
                *aten.detach.default
                *aten.detach.default
                *aten.view.default
                *aten.t.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.t.default
                *aten.addmm.default
                  shape: [torch.Size([10]), torch.Size([8, 16]), torch.Size([16, 10])]
                  sharding: [(Replicate(),), (Shard(dim=1),), (Shard(dim=0),)]
                *aten.div.Tensor
                *aten.addmm.default
                *_c10d_functional.all_reduce.default
                *aten.view.default
                *aten.sum.default
                *aten.ones_like.default

              BACKWARD PASS
                *aten.expand.default
                *aten.t.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.t.default
                *aten.mm.default
                  shape: [torch.Size([8, 10]), torch.Size([10, 16])]
                  sharding: [(Replicate(),), (Shard(dim=1),)]
                *aten.mm.default
                *aten.t.default
                  shape: [torch.Size([8, 10])]
                  sharding: [(Replicate(),)]
                *aten.t.default
                *aten.mm.default
                  shape: [torch.Size([10, 8]), torch.Size([8, 16])]
                  sharding: [(Replicate(),), (Shard(dim=1),)]
                *aten.mm.default
                *aten.t.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.t.default
                *aten.sum.dim_IntList
                  shape: [torch.Size([8, 10])]
                  sharding: [(Replicate(),)]
                *aten.sum.dim_IntList
                *aten.view.default
                  shape: [torch.Size([1, 10])]
                  sharding: [(Replicate(),)]
                *aten.view.default
                *aten.detach.default
                  shape: [torch.Size([10])]
                  sharding: [(Replicate(),)]
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([10])]
                  sharding: [(Replicate(),)]
                *aten.detach.default
                *aten.detach.default
                *aten.t.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.t.default
                *aten.detach.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.detach.default
                *aten.detach.default
                *aten.detach.default
                *aten.threshold_backward.default
                *aten.t.default
                  shape: [torch.Size([8, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.t.default
                *aten.mm.default
                  shape: [torch.Size([16, 8]), torch.Size([8, 10])]
                  sharding: [(Shard(dim=0),), (Replicate(),)]
                *aten.mm.default
                *aten.t.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.t.default
                *aten.sum.dim_IntList
                  shape: [torch.Size([8, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.sum.dim_IntList
                *aten.view.default
                  shape: [torch.Size([1, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.view.default
                *aten.detach.default
                  shape: [torch.Size([16])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([16])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                *aten.t.default
                  shape: [torch.Size([10, 16])]
                  sharding: [(Shard(dim=1),)]
                *aten.t.default
                *aten.detach.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
                  shape: [torch.Size([16, 10])]
                  sharding: [(Shard(dim=0),)]
                *aten.detach.default
                *aten.detach.default
        """

        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        inp_size = [8, 10]
        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)

        LR = 0.25

        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)

        comm_mode = CommDebugMode()

        with comm_mode:
            output_tp = model(inp)
            output_tp.sum().backward()

        # print the operation level collective tracing information
        print(comm_mode.generate_operation_tracing_table())
        comm_mode.log_operation_tracing_table_to_file()

    def test_transformer_operation_tracing(self, is_seq_parallel: bool = False) -> None:
        """
        Example code to demonstrate CommModeDebug's module operation level tracing using a distributed transformer model.
        Prints a table of module opoeration level collective tracing information and logs table to output.txt
        """
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        model_args = ModelArgs()
        model = Transformer(model_args).to(device=self.device_type)
        model = Transformer.parallelize(model, device_mesh, is_seq_parallel)

        LR = 0.25
        inp_size = [8, 8]

        torch.manual_seed(0)
        inp = torch.randint(model_args.vocab_size, inp_size, device=self.device_type)

        comm_mode = CommDebugMode()
        with comm_mode:
            output = model(inp)

        # print the operation level collective tracing information
        print(comm_mode.generate_operation_tracing_table())
        comm_mode.log_operation_tracing_table_to_file()


def run_example(world_size: int, rank: int, example_name: str) -> None:
    # set manual seed
    torch.manual_seed(0)
    # intializing class with all of the functions
    instantiated_test = CommDebugModeExample(world_size, rank)
    # dict that stores example code function names
    name_to_example_code: Dict[str, Callable[[], None]] = {
        "MLP_distributed_sharding_display": instantiated_test.test_MLP_distributed_sharding_display,
        "MLPStacked_distributed_sharding_display": instantiated_test.test_MLPStacked_distributed_sharding_display,
        "MLP_module_tracing": instantiated_test.test_MLP_module_tracing,
        "transformer_module_tracing": instantiated_test.test_transformer_module_tracing,
        "MLP_operation_tracing": instantiated_test.test_MLP_operation_tracing,
        "transformer_operation_tracing": instantiated_test.test_transformer_operation_tracing,
    }

    name_to_example_code[example_name]()


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4  # our example uses 4 worker ranks

    run_example(world_size, rank, args())
