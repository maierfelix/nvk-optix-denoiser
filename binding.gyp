{
  "variables": {
    "root": "../../..",
    "platform": "<(OS)",
    "release": "<@(module_root_dir)/build/Release",
    "vkSDK": "C:/VulkanSDK/1.1.121.2",
    "cudaSDK": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1",
    "optixSDK": "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
  },
  "conditions": [
    [ "platform == 'win'", { "variables": { "platform": "win" } } ]
  ],
  "targets": [
    {
      "target_name": "action_after_build",
      "type": "none",
      "conditions": []
    },
    {
      "sources": [
        "./src/index.cpp"
      ],
      "conditions": [
        [
          "OS=='win'",
          {
            "target_name": "addon-win32",
            "cflags": [
              "-stdlib=libstdc++"
            ],
            "include_dirs": [
              "<!@(node -p \"require('node-addon-api').include\")",
              "<(root)/lib/include/",
              "<(vkSDK)/include",
              "<(optixSDK)/include",
              "<(optixSDK)/SDK/cude",
              "<(optixSDK)/SDK",
              "<(optixSDK)/SDK/bin/include",
              "<(optixSDK)/SDK/bin",
              "<(cudaSDK)/include"
            ],
            "library_dirs": [
              "<(vkSDK)/lib",
              "<(vkSDK)/lib"
            ],
            "link_settings": {
              "libraries": [
                "-lvulkan-1.lib",
                "-l<(cudaSDK)/lib/x64/cudart_static.lib"
              ]
            },
            "defines": [
              "WIN32_LEAN_AND_MEAN",
              "VC_EXTRALEAN",
              "_ITERATOR_DEBUG_LEVEL=0",
              "_HAS_EXCEPTIONS=1"
            ],
            "msvs_settings": {
              "VCCLCompilerTool": {
                "FavorSizeOrSpeed": 1,
                "StringPooling": "true",
                "Optimization": 2,
                "WarningLevel": 3,
                "AdditionalOptions": ["/MP /EHsc"],
                "ExceptionHandling": 1
              },
              "VCLibrarianTool": {
                "AdditionalOptions" : ["/NODEFAULTLIB:MSVCRT"]
              },
              "VCLinkerTool": {
                "AdditionalLibraryDirectories": [
                  "../@PROJECT_SOURCE_DIR@/lib/<(platform)/<(target_arch)",
                ]
              }
            }
          }
        ]
      ]
    }
  ]
}