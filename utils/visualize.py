import PIL
import torch
from utils.vis import draw_scene_graph as original_draw_sg

COLORS = [
    (137, 49, 239),  # Blue-Violet
    (242, 202, 25),  # Yellow
    (255, 0, 189),  # Pink
    (0, 87, 233),  # Blue
    (135, 233, 17),  # Green
    (225, 24, 69),  # Orange
    (1, 176, 231),  # Cyan
    (138, 10, 183),  # Violet
    (138, 59, 59),  # Brown
]

HEXCOLORS = [
    '#8931EF',  # Blue-Violet
    '#F2CA19',  # Yellow
    '#FF00BD',  # Pink
    '#0057E9',  # Blue
    '#87E911',  # Green
    '#FF1845',  # Orange
    '#01B0E7',  # Cyan
    '#8A0AB7',  # Violet
    '#8A3B3B',  # Brown
]

def draw_scene_graph(graph, vocab):
    s, o = graph.edges()
    p = graph.edata['feat']
    triples = torch.stack((s, p, o), dim=1)
    sg_img = original_draw_sg(graph.ndata['feat'], triples, vocab)
    return PIL.Image.fromarray(sg_img)
