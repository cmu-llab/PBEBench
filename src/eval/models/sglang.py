import requests

def load_sglang_model(model_ckpt, tp=1):
    import sglang as sgl

    from sglang.srt.conversation import chat_templates
    from sglang.test.test_utils import is_in_ci
    from sglang.utils import async_stream_and_merge, stream_and_merge
    engine = sgl.Engine(model_path=model_ckpt, tp_size=tp, disable_custom_all_reduce=True)
    return engine

def generate_with_sglang_engine(engine,
                                input,
                                sample_parameters):
    output = engine.generate(input, sample_parameters)
    return output