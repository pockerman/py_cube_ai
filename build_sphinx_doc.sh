#sphinx-quickstart docs

#sphinx-apidoc -f -o docs/source docs/source/API
sphinx-build -b html docs/source/ docs/build/html
