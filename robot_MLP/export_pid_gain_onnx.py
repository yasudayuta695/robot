import argparse

import torch

from train import PIDGainMLP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export OpenCV-friendly PID gain ONNX from trained .pt state_dict")
    parser.add_argument("--weights-path", type=str, required=True, help="Path to pid gain model .pt")
    parser.add_argument("--onnx-path", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--kp-max", type=float, default=2.5)
    parser.add_argument("--ki-max", type=float, default=1.0)
    parser.add_argument("--kd-max", type=float, default=1.5)
    return parser


def main(args: argparse.Namespace) -> None:
    input_dim = int(args.history) * 9
    model = PIDGainMLP(
        input_dim=input_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        kp_max=args.kp_max,
        ki_max=args.ki_max,
        kd_max=args.kd_max,
    )

    state_dict = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to("cpu").eval()

    dummy = torch.randn(1, input_dim, dtype=torch.float32)
    export_kwargs = dict(
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["pid_gains"],
    )

    try:
        torch.onnx.export(model, dummy, args.onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, args.onnx_path, **export_kwargs)

    print(f"Exported OpenCV-friendly PID-gain ONNX: {args.onnx_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
