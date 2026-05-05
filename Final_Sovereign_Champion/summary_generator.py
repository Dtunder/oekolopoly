import os

class SummaryGenerator:
    """
    Generates an SVG summary of the Oekolopoly game state over time.
    Tracks Environment, Quality of Life, and Politics.
    """
    def __init__(self, state_history, translations=None):
        """
        :param state_history: List of np.array states
        :param translations: Dictionary of translations from translator.py
        """
        self.state_history = state_history
        self.dtl = translations if translations else {}

        # Indices from OekoEnv
        self.IDX_ENV = 5
        self.IDX_QOL = 3
        self.IDX_POL = 7
        self.IDX_ROUND = 8

    def generate_svg(self, output_path):
        if not self.state_history:
            return

        width = 800
        height = 600
        padding = 60

        chart_width = width - 2 * padding
        chart_height = height - 2 * padding

        max_rounds = 30
        max_v = 40  # Environment and QoL max is 29, Politics max is 37
        min_v = -10 # Politics can go down to -10

        v_range = max_v - min_v

        def scale_x(r):
            return padding + (r / max_rounds) * chart_width

        def scale_y(v):
            return height - padding - ((v - min_v) / v_range) * chart_height

        svg_lines = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="#0f141c" />', # Dark background to match GUI
            f'<text x="{width/2}" y="{padding/2}" fill="white" font-family="Arial" font-size="24" text-anchor="middle">Oekolopoly Game Summary</text>',

            # Axes
            f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="white" stroke-width="2" />',
            f'<line x1="{padding}" y1="{scale_y(0)}" x2="{width-padding}" y2="{scale_y(0)}" stroke="gray" stroke-width="1" stroke-dasharray="5,5" />', # Zero line
            f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="white" stroke-width="2" />',
        ]

        # Legend
        svg_lines.append(f'<text x="{padding}" y="{height-15}" fill="#40b464" font-family="Arial" font-size="14">{self.dtl.get("EnvirDamage", "Environment")}</text>')
        svg_lines.append(f'<text x="{padding+150}" y="{height-15}" fill="#3cb4f0" font-family="Arial" font-size="14">{self.dtl.get("QualityOfLife", "Quality of Life")}</text>')
        svg_lines.append(f'<text x="{padding+350}" y="{height-15}" fill="#e6be32" font-family="Arial" font-size="14">{self.dtl.get("Politics", "Politics")}</text>')

        # Data Polylines
        env_points = []
        qol_points = []
        pol_points = []

        for state in self.state_history:
            r = state[self.IDX_ROUND]
            env_points.append(f"{scale_x(r)},{scale_y(state[self.IDX_ENV])}")
            qol_points.append(f"{scale_x(r)},{scale_y(state[self.IDX_QOL])}")
            pol_points.append(f"{scale_x(r)},{scale_y(state[self.IDX_POL])}")

        svg_lines.append(f'<polyline points="{" ".join(env_points)}" fill="none" stroke="#40b464" stroke-width="3" />')
        svg_lines.append(f'<polyline points="{" ".join(qol_points)}" fill="none" stroke="#3cb4f0" stroke-width="3" />')
        svg_lines.append(f'<polyline points="{" ".join(pol_points)}" fill="none" stroke="#e6be32" stroke-width="3" />')

        # Grid labels
        for v in range(-10, 41, 10):
            y = scale_y(v)
            svg_lines.append(f'<text x="{padding-5}" y="{y+5}" fill="white" font-family="Arial" font-size="12" text-anchor="end">{v}</text>')

        for r in range(0, 31, 5):
            x = scale_x(r)
            svg_lines.append(f'<text x="{x}" y="{height-padding+20}" fill="white" font-family="Arial" font-size="12" text-anchor="middle">{r}</text>')

        svg_lines.append('</svg>')

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(svg_lines))
