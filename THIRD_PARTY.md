# Third-party components

The Apache-2.0 license in this repository covers the SE-ADD code authored by
the SE-ADD Authors. It does not relicense external models, repositories,
datasets, or services.

## Qwen2-Audio

SE-ADD can use `Qwen/Qwen2-Audio-7B-Instruct` as an external base model. Model
files are not included. Obtain them from the official Qwen distribution and
follow the license and acceptable-use terms published with that version:

- https://github.com/QwenLM/Qwen2-Audio
- https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

## MOSS-Audio

SE-ADD can use MOSS-Audio-8B-Instruct and imports its public Python interfaces.
Neither the MOSS-Audio repository nor its model weights are included. Obtain
them from the official project and follow the terms published with the exact
revision used:

- https://github.com/OpenMOSS/MOSS-Audio

At the revision used in our experiments, the project README states that
MOSS-Audio models are licensed under Apache License 2.0. Users should verify
the current upstream code and model terms themselves.

## Datasets

ASVspoof, HIR-SDD, In-the-Wild, and other evaluation audio are not included.
Names and small synthetic schema examples are provided solely to document the
interface. Users must obtain datasets from their official distributors and
comply with the corresponding licenses and access conditions.

## Python dependencies

Dependencies listed under `requirements/` retain their own licenses. Installing
them does not place them under the SE-ADD license.
