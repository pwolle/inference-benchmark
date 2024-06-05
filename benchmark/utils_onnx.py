import tempfile

import onnxruntime

# from . import performance
import performance
import torch


def from_torch(model, inputs):
    with tempfile.NamedTemporaryFile() as f:
        torch.onnx.export(model, inputs, f.name, verbose=False)
        return onnxruntime.InferenceSession(
            f.name,
            input_names=["input"],  # the model's input names
            output_names=["output"],
        )


def time_model_inference(
    model,
    *,
    args=None,
    kwargs=None,
    device="cpu",
):
    args = args or []
    kwargs = kwargs or {}

    model = from_torch(model, tuple(args))

    return performance.time_function_average(
        lambda *args, **kwargs: model.run(None, {"input": args[0].numpy()})[0],
        skip_first=True,
        args=args,
        kwargs=kwargs,
    )


def main():
    from models.vae_torch import get_vae_with_inputs

    vae, args, kwargs = get_vae_with_inputs(batch_size=32, device="cpu")
    time, times = time_model_inference(vae, args=args, kwargs=kwargs)
    print(time, len(times))


if __name__ == "__main__":
    main()
