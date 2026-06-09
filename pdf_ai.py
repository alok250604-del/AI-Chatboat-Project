import tkinter as tk
from tkinter import filedialog
from transformers import pipeline
import PyPDF2

print("Program start ho raha hai...")  # debug

# Summarizer
summarizer = pipeline("summarization")

# GUI
root = tk.Tk()
root.title("PDF AI Summarizer 📄")
root.geometry("600x600")

text_box = tk.Text(root)
text_box.pack(fill="both", expand=True)

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    
    if not file_path:
        return

    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, "Reading PDF...\n")

    pdf_text = read_pdf(file_path)

    text_box.insert(tk.END, "\nSummarizing...\n")

    summary = summarizer(pdf_text[:1000])

    text_box.insert(tk.END, "\nSummary:\n")
    text_box.insert(tk.END, summary[0]['summary_text'])

btn = tk.Button(root, text="Upload PDF", command=upload_file)
btn.pack()

print("GUI load ho raha hai...")  # debug

root.mainloop()