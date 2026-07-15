print("APP START")

from index import predict as model_predict

print("INDEX IMPORT SUCCESS")



import gradio as gr
import spaces
from index import predict as model_predict


@spaces.GPU
def predict(
    smiles_values,
    task,
    representation,
    model_name,
):
    if not isinstance(smiles_values, list):
        raise gr.Error("请输入由 SMILES 字符串组成的 JSON 数组")

    try:
        predictions = model_predict(
            smiles_values=smiles_values,
            task=task,
            representation=representation,
            model_name=model_name,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    return {
        "predictions": predictions,
        "meaning": {
            "0": "不稳定",
            "1": "稳定",
        },
    }


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.JSON(
            label="SMILES 数组",
            value=["CCO", "CCN"],
        ),
        gr.Dropdown(
            choices=["SIF", "SGF"],
            value="SIF",
            label="任务",
        ),
        gr.Dropdown(
            choices=["Morgan", "Avalon"],
            value="Morgan",
            label="分子表征",
        ),
        gr.Dropdown(
            choices=["lr", "rf", "xgb"],
            value="lr",
            label="预测模型",
        ),
    ],
    outputs=gr.JSON(label="预测结果"),
    title="SIF / SGF Stability Prediction",
    description="输入 SMILES 数组，0 表示不稳定，1 表示稳定。",
    api_name="predict",
)

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1
    ).launch(
        ssr_mode=False
    )