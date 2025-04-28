## Structure of the LLM Transparency Tool Repo
The code we implemented for the LLM Transparency Tool is split across 3 branches-
- main
- user/harshitgupta/layerskipCustomPoc
- user/harshitgupta/decoupledLayerskip

The main branch has modifications done by us for the following-
1. Added support to enable LLM Transparency tool on MacOS.
2. Enhanced hardware capability, by extending the system to utilize MPS GPUs too, in addition to existing support for CUDA.
3. programmaticLlmtt.ipynb : This notebook allows invoking the LLM transparency tool functionality programmatically, allowing us to not rely just on the UI for our inferences, enabling users to configure models, input prompts, thresholds, and visualize graphs programmatically.
4. Support for multiple LlaMA models, including meta-llama/Llama-3.2-1B.
5. Support for MBPP dataset using the custom mbppConverter.py script that allows us to extract problem descriptions from the MBPP dataset and flatten them into a plaintext format consumable by the tool.

The other two branches utilize the above changes and have support for the LayerSkip variants of Llama. 
layerskipCustomPoc: To support LayerSkip analysis, we created a custom subclass named LayerSkipLlamaTransparentLlm (in layerSkipCustom.py) that inherits from the base TransparentLlm interface used throughout LMTT. This class loads LayerSkip-based models using HuggingFace’s AutoModelForCausalLM, and defines all methods required by the interface.

decoupledLayerskip: Through our analysis we observed a tight coupling between the Server code and the TransformerLens class, which makes adding support for custom models non-trivial, since the server ends up invoking the methods of the TransformerLensTransparentLlm class instead of the custom class due to this tight coupling, unless we use a hacky approach defined in layerSkipCustomPoc branch. To move away from this hacky approach, this branch has our attempt at removing this tight decoupling between the existing classes and the server code.

## Instructions to setup LLM Transparency Tool

### Local Installation
#### MacOS
```bash
# download
git clone https://github.com/harshitgupta27/llmtt-experiments.git
cd llm-transparency-tool

# install the necessary packages
conda env create --name llmtt -f env.yaml
# install the `llm_transparency_tool` package
pip install -e .

# now, we need to build the frontend
cd llm_transparency_tool/components/frontend
yarn install
yarn build
```

#### Linux/ Windows
Replace env.yaml in our repo with this file: https://github.com/facebookresearch/llm-transparency-tool/blob/main/env.yaml
Follow the same instructions as of those in MacOS subsection

### Launch

```bash
streamlit run llm_transparency_tool/server/app.py -- config/local.json
```

## Using programmaticLlmtt.ipynb
1. Open the file in an IDE of your choice
2. Reuse the same conda environment created in the previous section.
3. The file can then be run cell by cell. We have included the code for loading models, building graphs, seeing its structure, as well as for computing average time of inferences.


# GENERIC INSTRUCTIONS GIVEN BY THE ORIGINAL AUTHORS OF THIS REPO


<h1>
  <img width="500" alt="LLM Transparency Tool" src="https://github.com/facebookresearch/llm-transparency-tool/assets/1367529/795233be-5ef7-4523-8282-67486cf2e15f">
</h1>

<img width="832" alt="screenshot" src="https://github.com/facebookresearch/llm-transparency-tool/assets/1367529/78f6f9e2-fe76-4ded-bb78-a57f64f4ac3a">


## Key functionality

* Choose your model, choose or add your prompt, run the inference.
* Browse contribution graph.
    * Select the token to build the graph from.
    * Tune the contribution threshold.
* Select representation of any token after any block.
* For the representation, see its projection to the output vocabulary, see which tokens
were promoted/suppressed but the previous block.
* The following things are clickable:
  * Edges. That shows more info about the contributing attention head.
  * Heads when an edge is selected. You can see what this head is promoting/suppressing.
  * FFN blocks (little squares on the graph).
  * Neurons when an FFN block is selected.


## Installation

### Dockerized running
```bash
# From the repository root directory
docker build -t llm_transparency_tool .
docker run --rm -p 7860:7860 llm_transparency_tool
```

### Local Installation


```bash
# download
git clone git@github.com:facebookresearch/llm-transparency-tool.git
cd llm-transparency-tool

# install the necessary packages
conda env create --name llmtt -f env.yaml
# install the `llm_transparency_tool` package
pip install -e .

# now, we need to build the frontend
# don't worry, even `yarn` comes preinstalled by `env.yaml`
cd llm_transparency_tool/components/frontend
yarn install
yarn build
```

### Launch

```bash
streamlit run llm_transparency_tool/server/app.py -- config/local.json
```


## Adding support for your LLM

Initially, the tool allows you to select from just a handful of models. Here are the
options you can try for using your model in the tool, from least to most
effort.


### The model is already supported by TransformerLens

Full list of models is [here](https://github.com/neelnanda-io/TransformerLens/blob/0825c5eb4196e7ad72d28bcf4e615306b3897490/transformer_lens/loading_from_pretrained.py#L18).
In this case, the model can be added to the configuration json file.


### Tuned version of a model supported by TransformerLens

Add the official name of the model to the config along with the location to read the
weights from.


### The model is not supported by TransformerLens

In this case the UI wouldn't know how to create proper hooks for the model. You'd need
to implement your version of [TransparentLlm](./llm_transparency_tool/models/transparent_llm.py#L28) class and alter the
Streamlit app to use your implementation.

## Citation
If you use the LLM Transparency Tool for your research, please consider citing:

```bibtex
@article{tufanov2024lm,
      title={LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models}, 
      author={Igor Tufanov and Karen Hambardzumyan and Javier Ferrando and Elena Voita},
      year={2024},
      journal={Arxiv},
      url={https://arxiv.org/abs/2404.07004}
}

@article{ferrando2024information,
    title={Information Flow Routes: Automatically Interpreting Language Models at Scale}, 
    author={Javier Ferrando and Elena Voita},
    year={2024},
    journal={Arxiv},
    url={https://arxiv.org/abs/2403.00824}
}
````

## License

This code is made available under a [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license, as found in the LICENSE file.
However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
