### Install RhodesLeRobot 🤗

#### From Source

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/TeleHuman/RhodesLeRobot.git
cd RhodesLeRobot
```

Then, install the library in editable mode. This is useful if you plan to contribute to the code.

```bash
pip install -e .
```

#### Fine-tuning PI0 and PI0.5
Note that if you want to fine-tune pi0 and pi0.5, you should instead run

```bash
pip install -e '.[pi]'
```