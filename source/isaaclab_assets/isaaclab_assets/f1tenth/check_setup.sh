#!/bin/bash
# Quick setup checker for F1TENTH environment

echo "================================================"
echo "F1TENTH Isaac Lab Setup Checker"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: USD file
echo -n "1. Checking USD file... "
if [ -f "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth/f1tenth.usd" ]; then
    echo -e "${GREEN}✓ Found${NC}"
    ls -lh /workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth/f1tenth.usd
else
    echo -e "${RED}✗ Missing${NC}"
    echo -e "${YELLOW}   → Run: cd source/isaaclab_assets/isaaclab_assets/f1tenth && ./convert_to_usd.sh${NC}"
fi

echo ""

# Check 2: Environment file
echo -n "2. Checking environment file... "
if [ -f "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/f1tenth/f1tenth_env.py" ]; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Missing${NC}"
fi

echo ""

# Check 3: Agent config
echo -n "3. Checking SAC config... "
if [ -f "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/f1tenth/agents/skrl_sac_cfg.yaml" ]; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Missing${NC}"
fi

echo ""

# Check 4: Registration
echo -n "4. Checking environment registration... "
if grep -q "f1tenth" /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py 2>/dev/null; then
    echo -e "${GREEN}✓ Registered${NC}"
else
    echo -e "${RED}✗ Not registered${NC}"
fi

echo ""
echo "================================================"
echo "File Structure:"
echo "================================================"
tree -L 2 /workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth 2>/dev/null || \
    ls -lah /workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth

echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo "1. Test environment:"
echo "   ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/f1tenth/test_f1tenth.py --num_envs 4 --headless"
echo ""
echo "2. Train with SAC:"
echo "   ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-F1tenth-Direct-v0 --headless"
echo ""
