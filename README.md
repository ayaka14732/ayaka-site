# ayaka-site ![](https://github.com/ayaka14732/ayaka-site/workflows/Publish/badge.svg?branch=main)

## Build (local)

Install prerequisites:

```sh
sudo pacman -S pandoc
```

Build HTML:

```sh
python build.py
```

Build PDF (experimental):

```sh
./build-pdf.sh dir
```

## Build (server)

See the [publish script](https://github.com/ayaka14732/ayaka-site/blob/main/.github/workflows/publish.yml).
