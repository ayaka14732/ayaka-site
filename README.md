# ayaka-site

Ayaka’s personal website

## The site

The site is deployed at [ayaka.shn.hk](https://ayaka.shn.hk/).

## Site contents

See the [main branch](https://github.com/ayaka14732/ayaka-site/tree/main).

## Build

### Local

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

### Server

See the [publish script](https://github.com/ayaka14732/ayaka-site/blob/main/.github/workflows/publish.yml).
