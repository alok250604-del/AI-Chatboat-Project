import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 🤖 Smart FREE model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# 🎨 GUI setup
root = tk.Tk()
root.title("AI Chatbot 🤖")
root.geometry("500x600")

chat_box = tk.Text(root, bg="white", font=("Arial", 12))
chat_box.pack(padx=10, pady=10, fill="both", expand=True)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(padx=10, pady=5, fill="x")

# 🚀 Chat function
def send():
    user_input = entry.get().strip()
    if not user_input:
        return

    chat_box.insert(tk.END, "You: " + user_input + "\n")

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(
        output[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    if reply.strip() == "":
        reply = "I didn't understand that properly."

    chat_box.insert(tk.END, "AI: " + reply + "\n\n")
    entry.delete(0, tk.END)

# 🔘 Button
btn = tk.Button(root, text="Send 🚀", command=send)
btn.pack()

# ⌨️ Enter key support
root.bind("<Return>", lambda event: send())

# ▶️ Run app
root.mainloop()