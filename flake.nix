{
  description = "Interactive C-RADIOv4 dance analysis workbench";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            python312Packages.pip
            uv
            bun
            ffmpeg
            pkg-config
            stdenv.cc.cc.lib
          ];

          shellHook = ''
            export UV_PROJECT_ENVIRONMENT=.venv
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}:$LD_LIBRARY_PATH
          '';
        };
      });
}
