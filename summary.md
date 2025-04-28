## Structure of the LLM Transparency Tool Repo
The code we implemented for the LLM Transparency Tool is split across 3 branches-
- main
- user/harshitgupta/layerskipCustomPoc
- user/harshitgupta/decoupledLayerskip

The main branch has modifications done by us for the following-
1. Added support to Enable LLM Transparency tool to be set up on MacOS.
1. Enhanced hardware capability, by extending the system to utilize MPS GPUs too, in addition to existing support for CUDA.
2. programmaticLlmtt.py : This notebook allows invoking the LLM transparency tool functionality programmatically, allowing us to not rely just on the UI for our inferences, enabling users to configure models, input prompts, thresholds, and visualize graphs programmatically.
3. Support for multiple LlaMA models, including meta-llama/Llama-3.2-1B.
4. Support for MBPP dataset using the custom mbppConverter.py script that allows us to extract problem descriptions from the MBPP dataset and flatten them into a plaintext format consumable by the tool.

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

## Using programmaticLlmtt.ipynb:
1. Open the file in an IDE of your choice
2. Reuse the same conda environment created in the previous section.
3. The file can then be run cell by cell. We have included the code for loading models, building graphs, seeing its structure, as well as for computing average time of inferences.

