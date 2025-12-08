# Gradio Web Application for Legal Text Decoder
# Provides a user-friendly interface for analyzing legal text understandability.

import os
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoTokenizer

from model import LegalTextClassifier
import config

# Label information
LABEL_INFO = {
    1: {"name": "Nagyon nehezen érthető", "english": "Very hard to understand", "color": "#ff4444"},
    2: {"name": "Nehezen érthető", "english": "Hard to understand", "color": "#ff8844"},
    3: {"name": "Többé/kevésbé megértem", "english": "Somewhat understandable", "color": "#ffcc44"},
    4: {"name": "Érthető", "english": "Understandable", "color": "#88cc44"},
    5: {"name": "Könnyen érthető", "english": "Easy to understand", "color": "#44cc44"}
}


class LegalTextApp:
    """Gradio application for legal text analysis."""

    def __init__(self, model_path: str = "models/best_model.pth"):
        """Initialize the application with the trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        # Try to load model
        if Path(model_path).exists():
            self._load_model(model_path)
        else:
            print(f"Warning: Model not found at {model_path}. Using demo mode.")

    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            model_config = checkpoint.get('config', {})
            model_name = model_config.get('model_name', config.MODEL_NAME)
            num_labels = model_config.get('num_labels', config.NUM_LABELS)

            self.model = LegalTextClassifier(
                model_name=model_name,
                num_labels=num_labels
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_length = model_config.get('max_length', config.MAX_LENGTH)
            self.model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def predict(self, text: str) -> tuple:
        """
        Predict understandability of input text.

        Returns:
            Tuple of (label_display, confidence_display, probability_chart_data)
        """
        if not text or not text.strip():
            return "Please enter some text to analyze.", "", {}

        if not self.model_loaded:
            # Demo mode - return random-ish results based on text length
            import random
            random.seed(len(text))
            label = random.randint(1, 5)
            confidence = random.uniform(0.4, 0.9)
            probs = {f"Label {i}": random.uniform(0.05, 0.3) for i in range(1, 6)}
            probs[f"Label {label}"] = confidence

            label_info = LABEL_INFO[label]
            label_display = f"**{label}** - {label_info['name']} ({label_info['english']})"
            confidence_display = f"Confidence: {confidence:.1%}"

            return label_display, confidence_display, probs

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            predicted_class = probs.argmax()

        # Convert to 1-indexed label
        label = predicted_class + 1
        confidence = probs[predicted_class]

        # Format outputs
        label_info = LABEL_INFO[label]
        label_display = f"**{label}** - {label_info['name']} ({label_info['english']})"
        confidence_display = f"Confidence: {confidence:.1%}"

        # Create probability dict for chart
        prob_dict = {
            f"{i}: {LABEL_INFO[i]['name'][:15]}...": float(probs[i-1])
            for i in range(1, 6)
        }

        return label_display, confidence_display, prob_dict


def create_app(model_path: str = "models/best_model.pth"):
    """Create and configure the Gradio application."""
    app = LegalTextApp(model_path)

    # Example texts
    examples = [
        ["A vásárló a megrendelt termék átvételére 14 napon belül köteles."],
        ["Az üzletszabályzat módosítása esetén az e tárgyban közzétett hirdetmény tartalmazza a megváltoztatott szabályokat és a hatálybalépés napját is."],
        ["A Vtv. 69. § 25. pontja alapján a vasúti társaság Üzletszabályzatának jóváhagyása a Vtv. szerinti vasúti igazgatási szerv hatáskörébe tartozik."],
        ["Fenntartjuk a jogot, hogy ne fogadjunk el, vagy töröljünk bármely megrendelést, ha azt a megrendelést szoftver, kereső vagy bármilyen más robot illetve bármilyen automatizált rendszer vagy szkript, vagy az Ön nevében történő megrendelés leadására használt harmadik fél szolgáltatásainak használatával adják le vagy hajtják végre."],
        ["A szolgáltatás díja havi 2000 Ft."],
    ]

    # Create Gradio interface
    with gr.Blocks(
        title="Legal Text Decoder",
        theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("""
        # Legal Text Decoder
        ## Magyar jogi szövegek érthetőségének elemzése

        Ez az alkalmazás magyar jogi szövegek (ÁSZF-ek, szerződések) érthetőségét elemzi
        egy 1-5 skálán, ahol:
        - **1** = Nagyon nehezen érthető
        - **5** = Könnyen érthető

        Írjon be vagy másoljon be egy jogi szöveget az elemzéshez.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Jogi szöveg / Legal Text",
                    placeholder="Írja be vagy másolja ide a jogi szöveget...",
                    lines=8,
                    max_lines=20
                )
                analyze_btn = gr.Button("Elemzés / Analyze", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Markdown(label="Érthetőségi szint / Understandability")
                confidence_output = gr.Markdown(label="Bizonyosság / Confidence")
                prob_chart = gr.Label(
                    label="Valószínűségek / Probabilities",
                    num_top_classes=5
                )

        gr.Examples(
            examples=examples,
            inputs=text_input,
            label="Példák / Examples"
        )

        # Connect components
        analyze_btn.click(
            fn=app.predict,
            inputs=text_input,
            outputs=[label_output, confidence_output, prob_chart]
        )

        text_input.submit(
            fn=app.predict,
            inputs=text_input,
            outputs=[label_output, confidence_output, prob_chart]
        )

        gr.Markdown("""
        ---
        ### About

        This model was trained on Hungarian legal texts (ÁSZF - General Terms and Conditions)
        using a fine-tuned Hungarian BERT model (HuBERT).

        **Model**: SZTAKI-HLT/hubert-base-cc

        **Dataset**: 3,397 labeled paragraphs from various Hungarian legal documents
        """)

    return interface


def main():
    """Launch the Gradio application."""
    interface = create_app()
    interface.launch(
        server_name=config.API_HOST,
        server_port=config.GRADIO_PORT,
        share=False
    )


if __name__ == "__main__":
    main()
