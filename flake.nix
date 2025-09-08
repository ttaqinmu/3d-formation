{
  description = "Multi Agent Quadcopter Simulation";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      python = pkgs.python312;
      pythonPackages = pkgs.python312Packages;
      xorgLibs = with pkgs.xorg; [
        libICE
        libSM
        libX11
        libX11.dev
      ];
      extraLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        libGL
        libglvnd
        glfw
      ] ++ xorgLibs;
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python.withPackages (ps: with ps; [
            pip
          ]))

          pyright

          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl

          gcc
        ] ++ extraLibs;

        shellHook = ''
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath extraLibs}
          export CUDA_HOME=${pkgs.cudatoolkit}
          export PYTORCH_NO_CUDA_MEMORY_CACHING=1
          export PYTORCH_NO_CUDA_BUILD=1

          source .venv/bin/activate
        '';
      };
    };
}
