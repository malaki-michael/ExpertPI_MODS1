import os
from pathlib import Path


def get_assets_path():
    path = Path(__file__).parent.parent / 'docs' / 'assets'
    return str(path)


def get_docs():
    docs = {}
    path = os.path.join(os.path.dirname(__file__), '..', 'docs')
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read().splitlines()
                    content = [line.replace("(assets/", f"({get_assets_path()}/") for line in content]
                docs[file] = content
    return docs


if __name__ == '__main__':
    docs = get_docs()
    print(docs)
