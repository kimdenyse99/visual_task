# 2ac-vocalization-discriminate

Behavioral experiment for two-alternative-choice vocalization discrimination.

## Setup

This private repository includes `py-virtualcall` as a private Git submodule.
Access is limited to the existing authorized internal group; do not publish or
redistribute the repository or its submodule.

For a new checkout, clone with all submodules and then sync the locked
environment:

```bash
git clone --recurse-submodules <repository-url>
cd 2ac-vocalization-discriminate
uv sync --locked
```

For an existing checkout, initialize the submodule before syncing:

```bash
git submodule update --init --recursive
uv sync --locked
```

## Updating `py-virtualcall`

The submodule is pinned to the commit recorded by this repository. To upgrade
it, explicitly check out the reviewed commit inside `vendor/py-virtualcall`,
then commit the updated gitlink in this repository. Do not configure automatic
upstream branch tracking.

Standard source archives do not include submodule contents and are not an
offline delivery format. Use a recursive Git clone with authorized access.
