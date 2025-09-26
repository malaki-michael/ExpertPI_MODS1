import os


def get_docs():
    docs = {}
    path = os.path.join(os.path.dirname(__file__), '..', 'docs')
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                with open(os.path.join(root, file), 'r') as f:
                    docs[file] = f.read().splitlines()
                    
    return docs


if __name__ == '__main__':
    docs = get_docs()
    print(docs)
