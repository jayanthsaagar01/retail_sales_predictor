# Environment configuration for Retail Forecaster
# This replaces the replit.nix configuration

{ pkgs }:

{
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.wheel
    pkgs.python311Packages.setuptools
  ];
}