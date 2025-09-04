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
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python.withPackages (ps: with ps; [
            pip
            # numpy
            # matplotlib
            # numba
            # pybullet
            # pettingzoo
            # torchWithCuda
            # torchrl
            # rerun-sdk
          ]))
          
          pyright
          stdenv.cc.cc.lib
	  
	  # rerun

          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl

          gcc
          zlib
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
          export CUDA_HOME=${pkgs.cudatoolkit}
          export PYTORCH_NO_CUDA_MEMORY_CACHING=1
          export PYTORCH_NO_CUDA_BUILD=1

          source .venv/bin/activate
        '';
      };
    };
}
