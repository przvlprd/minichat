from llama_cpp import Llama
from settings import model_path, n_ctx, n_gpu_layers, n_batch

llm = Llama(
    model_path=model_path,
    n_ctx=n_ctx,  # 4096?
    last_n_tokens_size=256,
    # n_threads=4,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    verbose=False  # to avoid any model info output
)

while True:
    prompt = f"""
    ### HUMAN:
    {input()}
    
    ### RESPONSE:
    """

    # With streaming
    stream = llm.create_completion(
        prompt,
        stream=True,
        repeat_penalty=1.1,
        max_tokens=256,
        stop=["USER:", "ASSISTANT:", "HUMAN", "RESPONSE", "###"],
        echo=False,

        temperature=0.3,
        mirostat_mode=2,
        mirostat_tau=4.0,
        mirostat_eta=1.1)

    result = ""
    for output in stream:
        result += output['choices'][0]['text']

    print(result)


# ToDo:
#  - keep context / memory
#   - summarise chat history, keep as context?
#   - see jupyter nb DLAI
#  - LangChain integration
#  - serve as API
#   - TG bot?
