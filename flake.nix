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
            uv
          ]))

          pyright

          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.nccl

          gcc
        ] ++ extraLibs;

        shellHook = ''
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath extraLibs}:${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.nccl}/lib:/run/opengl-driver/lib
          export CUDA_HOME=${pkgs.cudatoolkit}

          source .venv/bin/activate
        '';
      };
    };
}
