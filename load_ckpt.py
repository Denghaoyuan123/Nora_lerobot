from transformers import AutoProcessor, AutoModelForVision2Seq

# 设置保存路径
model_dir = "/data4/hydeng/nora/ckpt/nora-long/"
# model_dir = "/home/e230112/nora/ckpt/nora-libero-spatial/"


# 下载并保存 processor 和 model 到指定目录
processor = AutoProcessor.from_pretrained("declare-lab/nora-long", cache_dir=model_dir)
model = AutoModelForVision2Seq.from_pretrained("declare-lab/nora-long", cache_dir=model_dir)


# processor = AutoProcessor.from_pretrained("declare-lab/nora-finetuned-libero-spatial", cache_dir=model_dir)
# model = AutoModelForVision2Seq.from_pretrained("declare-lab/nora-finetuned-libero-spatial", cache_dir=model_dir)