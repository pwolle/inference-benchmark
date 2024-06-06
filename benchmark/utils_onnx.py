import tempfile

import onnxruntime

from . import performance
import torch


def from_torch(model, inputs):
    with tempfile.NamedTemporaryFile() as f:
        torch.onnx.export(
            model,
            inputs,
            f.name,
            input_names=["input"],
            output_names=["output"],
        )
        opts = onnxruntime.SessionOptions()
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        return onnxruntime.InferenceSession(
            f.name,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )


def time_model_inference(model, x: torch.Tensor):
    model = from_torch(model, x)
    x = x.detach().cpu().numpy()

    return performance.time_function_average(
        lambda x: model.run(None, {"input": x}),
        skip_first=True,
        args=(x,),
    )


def main():
    from models.vae_torch import get_vae_with_inputs

    vae, args, kwargs = get_vae_with_inputs(batch_size=32, device="cpu")
    time, times = time_model_inference(vae, args=args, kwargs=kwargs)
    print(time, len(times))


if __name__ == "__main__":
    main()
