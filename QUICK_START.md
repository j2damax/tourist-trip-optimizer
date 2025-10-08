# Quick Start Guide

**New to optimization?** Get started in 15 minutes!

## Option 1: Interactive Tutorial (Recommended for Beginners)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the beginner tutorial
cd examples
python beginner_tutorial.py

# 3. Open the interactive notebook
jupyter notebook ../notebooks/05_Interactive_Learning_Tutorial.ipynb
```

**Time needed**: 15-30 minutes  
**What you'll learn**: How GA works, step by step

## Option 2: Read the Learning Guide

Open [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) and start with:
- Section 1: Introduction to Optimization (10 min)
- Section 2: Core Concepts (20 min)
- Section 5: Step-by-Step Genetic Algorithm (30 min)

**Time needed**: 1 hour  
**What you'll learn**: Theory and concepts behind optimization

## Option 3: Jump Right In

```bash
# Run the full GA implementation
jupyter notebook notebooks/02_Genetic_Algorithm_Implementation.ipynb
```

**Time needed**: 5 minutes to run  
**Best for**: Learning by experimentation

## Learning Path

```
Day 1 (2 hours):
├── Read LEARNING_GUIDE.md sections 1-2
├── Run examples/beginner_tutorial.py
└── Try exercises in section 8

Day 2 (2 hours):
├── Read LEARNING_GUIDE.md sections 5-6
├── Run notebooks/05_Interactive_Learning_Tutorial.ipynb
└── Experiment with parameters

Day 3 (2 hours):
├── Run notebooks/02_Genetic_Algorithm_Implementation.ipynb
├── Run notebooks/03_MIP_Model_Benchmark.ipynb
└── Compare results

Week 2-3:
└── Follow the 3-week roadmap in LEARNING_GUIDE.md section 9
```

## Common Questions

**Q: I have no optimization background. Where do I start?**  
A: Start with `examples/beginner_tutorial.py`, then read LEARNING_GUIDE.md section 1-2.

**Q: What's the difference between GA and MIP?**  
A: GA is fast and finds good solutions. MIP is slower but finds optimal solutions. See LEARNING_GUIDE.md section 7.

**Q: Can I use this for my own problem?**  
A: Yes! See LEARNING_GUIDE.md section 8 (exercises) for examples.

**Q: How long does it take to learn?**  
A: Basics: 1 week. Proficiency: 2-3 weeks. Mastery: 2-3 months of practice.

## Resources at a Glance

| Resource | Purpose | Time | Difficulty |
|----------|---------|------|------------|
| `examples/beginner_tutorial.py` | First hands-on experience | 15 min | ⭐ Beginner |
| `notebooks/05_Interactive_Learning_Tutorial.ipynb` | Step-by-step learning | 60 min | ⭐ Beginner |
| `LEARNING_GUIDE.md` | Complete educational guide | 10-20 hrs | ⭐⭐ Beginner-Intermediate |
| `notebooks/02_Genetic_Algorithm_Implementation.ipynb` | Full GA on real data | 30 min | ⭐⭐ Intermediate |
| `notebooks/03_MIP_Model_Benchmark.ipynb` | MIP optimization | 30 min | ⭐⭐⭐ Advanced |
| `TECHNICAL.md` | Deep technical details | 3-5 hrs | ⭐⭐⭐ Advanced |

## Need Help?

1. Check LEARNING_GUIDE.md section 10 (Common Pitfalls)
2. Read the relevant notebook's markdown cells
3. Open an issue on GitHub
4. Check LEARNING_GUIDE.md section 3 for external resources

## Next Steps

After completing the basics:
- [ ] Read LEARNING_GUIDE.md sections 7-11
- [ ] Complete all exercises in section 8
- [ ] Try the challenges in the interactive notebook
- [ ] Modify the code to solve your own problem
- [ ] Read papers in LEARNING_GUIDE.md section 11

---

**Ready to start?** Run: `python examples/beginner_tutorial.py` 🚀
