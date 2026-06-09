import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ⚡ Small + fast model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# 🎨 GUI
root = tk.Tk()
root.title("Fast AI Chatbot ⚡")
root.geometry("500x600")

chat_box = tk.Text(root, bg="white", font=("Arial", 12))
chat_box.pack(padx=10, pady=10, fill="both", expand=True)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(padx=10, pady=5, fill="x")

# 🚀 Send function (FAST MODE)
def send():
    user_input = entry.get().strip()
    if not user_input:
        return

    chat_box.insert(tk.END, "You: " + user_input + "\n")

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    with torch.no_grad():  # ⚡ speed boost
        output = model.generate(
            input_ids,
            max_new_tokens=40,   # ⚡ shorter output = faster
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    chat_box.insert(tk.END, "AI: " + reply + "\n\n")
    entry.delete(0, tk.END)

# 🔘 Button
btn = tk.Button(root, text="Send ⚡", command=send)
btn.pack()

# ⌨️ Enter key
root.bind("<Return>", lambda e: send())

root.mainloop()