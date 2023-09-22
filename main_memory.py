from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from settings import model_path, n_ctx, n_gpu_layers, n_batch

llm = LlamaCpp(
    model_path=model_path,
    n_ctx=n_ctx,
    last_n_tokens_size=1024,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    verbose=False,

    stop=["USER:", "ASSISTANT:", "HUMAN", "RESPONSE", "###", "AI:",
          "Human:", "\n"],
    repeat_penalty=1.1,
    max_tokens=512,
    temperature=0.3,
)


memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

while True:
    # print(conversation.predict(input=input("User: ")))
    print(f"AI: {conversation.predict(input=input('User: '))} ")
