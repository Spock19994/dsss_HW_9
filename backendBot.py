from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(update)
    await update.message.reply_text("Received your message!")
# Command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("Received /start command")
    await update.message.reply_text("Hallo! I'm your AI Assistant bot. How can I help you?")

# Message
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text.strip()
    print(f"User input: {user_input}")
    messages = [
        {"role": "system", "content": "You are a friendly chatbot"},
        {"role": "user", "content": user_input},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"].split('<|assistant|>')[-1].strip()
    await update.message.reply_text(response)

def main():
    application = Application.builder().token("8004913488:AAE-EA7sc0-eXPtr3t5Od-Pssy4Dx1_PNlc").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(MessageHandler(filters.TEXT, log_update))
    application.run_polling()

if __name__ == "__main__":
    main()
