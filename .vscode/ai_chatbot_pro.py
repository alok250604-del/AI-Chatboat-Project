import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 🤖 Model load (smart + free)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

# 🎨 GUI setup
root = tk.Tk()
root.title("AI Chatbot PRO 🤖")
root.geometry("500x650")

chat_box = tk.Text(root, bg="white", font=("Arial", 12))
chat_box.pack(padx=10, pady=10, fill="both", expand=True)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(padx=10, pady=5, fill="x")

# 💾 Save chat function
def save_chat(text):
    with open("history.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

# 🚀 Send function
def send():
    global chat_history_ids

    user_input = entry.get().strip()
    if not user_input:
        return

    chat_box.insert(tk.END, "You: " + user_input + "\n")
    save_chat("You: " + user_input)

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    with torch.no_grad():
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=500,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    if reply.strip() == "":
        reply = "I didn't understand that properly."

    chat_box.insert(tk.END, "AI: " + reply + "\n\n")
    save_chat("AI: " + reply)

    entry.delete(0, tk.END)

# 🧹 Clear chat
def clear_chat():
    global chat_history_ids
    chat_box.delete("1.0", tk.END)
    chat_history_ids = None

# 🔘 Buttons
frame = tk.Frame(root)
frame.pack(pady=5)

send_btn = tk.Button(frame, text="Send 🚀", command=send)
send_btn.grid(row=0, column=0, padx=5)

clear_btn = tk.Button(frame, text="Clear 🧹", command=clear_chat)
clear_btn.grid(row=0, column=1, padx=5)

# ⌨️ Enter key support
root.bind("<Return>", lambda event: send())

# ▶️ Run app
root.mainloop()