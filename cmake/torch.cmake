execute_process(
  COMMAND bash -c
  "python -c 'import torch;print(torch.utils.cmake_prefix_path)'"
  OUTPUT_VARIABLE SHELL_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(TORCH_CMAKE_PREFIX_PATH ${SHELL_OUTPUT})
