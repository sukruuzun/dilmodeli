#!/bin/bash

# DilModeli Demo Çalıştırma Scripti

echo "======================================"
echo "  DilModeli Demo Scripti"
echo "======================================"
echo ""

# Renk kodları
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Demo seçenekleri
echo "Hangi demo'yu çalıştırmak istersiniz?"
echo ""
echo "1) Nash-Sürü MoE Demo"
echo "2) Dinamik Kuantizasyon ve Budama Demo"
echo "3) CPU Optimizasyon Demo"
echo "4) Tam Sistem Demo"
echo "5) Tümünü Çalıştır"
echo "0) Çıkış"
echo ""

read -p "Seçiminiz (0-5): " secim

case $secim in
    1)
        echo -e "${BLUE}Nash-Sürü MoE Demo çalıştırılıyor...${NC}"
        python3 examples/demo_nash_suru_moe.py
        ;;
    2)
        echo -e "${BLUE}Kuantizasyon Demo çalıştırılıyor...${NC}"
        python3 examples/demo_kuantizasyon.py
        ;;
    3)
        echo -e "${BLUE}CPU Optimizasyon Demo çalıştırılıyor...${NC}"
        python3 examples/demo_cpu_optimizer.py
        ;;
    4)
        echo -e "${BLUE}Tam Sistem Demo çalıştırılıyor...${NC}"
        python3 examples/demo_tam_sistem.py
        ;;
    5)
        echo -e "${YELLOW}Tüm demo'lar sırayla çalıştırılıyor...${NC}"
        echo ""
        
        echo -e "${BLUE}1/4: Nash-Sürü MoE Demo${NC}"
        python3 examples/demo_nash_suru_moe.py
        echo ""
        read -p "Devam etmek için Enter'a basın..."
        
        echo -e "${BLUE}2/4: Kuantizasyon Demo${NC}"
        python3 examples/demo_kuantizasyon.py
        echo ""
        read -p "Devam etmek için Enter'a basın..."
        
        echo -e "${BLUE}3/4: CPU Optimizasyon Demo${NC}"
        python3 examples/demo_cpu_optimizer.py
        echo ""
        read -p "Devam etmek için Enter'a basın..."
        
        echo -e "${BLUE}4/4: Tam Sistem Demo${NC}"
        python3 examples/demo_tam_sistem.py
        
        echo ""
        echo -e "${GREEN}Tüm demo'lar tamamlandı!${NC}"
        ;;
    0)
        echo -e "${GREEN}Çıkış yapılıyor...${NC}"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Geçersiz seçim! Lütfen 0-5 arası bir sayı girin.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Demo tamamlandı!${NC}"
echo ""

