"""Gradio demo:  python app.py   (works offline once models/final.pt exists)"""
import gradio as gr
from predict import predict_path


def run(path):
    if path is None: return "Upload an image", ""
    r = predict_path(path)
    return r["label"] + (f"  (p = {r['prob']:.2f})" if r["prob"] is not None else ""), r["message"]


gr.Interface(run, gr.Image(type="filepath", label="Paediatric chest X-ray"),
             [gr.Textbox(label="Result"), gr.Textbox(label="Guidance")],
             title="MediScan (Arogya Diagnostics) - screening aid, NOT a diagnosis",
             description="Trained on 1-5 year-olds, AP view, single hospital. Not validated for adults, other views or other scanners.").launch()
