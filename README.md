# Human-like Cognition Prototype

This project is a modular cognitive architecture inspired by how humans process information: perception, attention, memory, emotion, planning, decision-making, learning, and self-reflection.

It is intentionally designed as a clear educational prototype rather than a full brain model. The goal is to show how these cognitive systems can work together in one agent.

## Features

- Perception: sensory input processing
- Attention: salience weighting based on concepts, memory, and emotion
- Working memory: short-term event buffer
- Long-term memory: concept reinforcement and episodic memory
- Emotion: valence and arousal updates from outcomes
- Planning: goal generation and action planning
- Decision making: rule-based or Q-learning-based selection
- Learning: reward-driven updates
- Self-reflection: adaptive changes based on recent success/failure

## Project structure

- `cognition/` — cognitive modules
- `demo.py` — example run of the agent
- `requirements.txt` — Python dependencies
- `README.md` — project overview

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

## Notes

The project uses a lightweight fallback pipeline so it can run without deep learning libraries. If PyTorch is available, the perception module can use a tiny neural embedding model.

## License

MIT
