@echo off
echo ========================================
echo  v4 Ablation Study: A B C D AC ABC ABCD
echo ========================================

set PYTHON=/c/Users/savyc/miniconda3/python.exe

for %%v in (base A B C D AC ABC ABCD) do (
    echo.
    echo ===== Training variant: %%v =====
    set VARIANT=%%v
    %PYTHON% train_v4_ablation.py
)

echo.
echo ===== All training complete. Running tests =====

for %%v in (base A B C D AC ABC ABCD) do (
    echo.
    echo ===== Testing variant: %%v =====
    set VARIANT=%%v
    %PYTHON% test_v4_ablation.py
)

echo.
echo ===== Ablation study complete =====
pause
