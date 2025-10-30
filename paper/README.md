# Nash-Swarm Optimization Paper

Bu klasör ArXiv paper'ı için gerekli dosyaları içeriyor.

## Dosyalar

- `main.tex` - Ana paper dosyası (LaTeX)
- `references.bib` - Kaynaklar (BibTeX)
- `figures/` - Görseller ve tablolar (oluşturulacak)
- `sections/` - Bölümler (opsiyonel, organize etmek için)

## Compile Etme

### Overleaf (Tavsiye Edilen)
1. https://overleaf.com → New Project → Upload Project
2. Bu klasördeki dosyaları upload et
3. "Recompile" tıkla
4. PDF indir

### Yerel (MacTeX/TexLive)
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Paper Yapısı

- **Abstract**: 150-200 kelime ✅ (taslak hazır)
- **Introduction**: 1 sayfa (yazılacak)
- **Related Work**: 1 sayfa (yazılacak)
- **Method**: 2 sayfa (yazılacak)
- **Experiments**: 2 sayfa (yazılacak)
- **Discussion**: 1 sayfa (yazılacak)
- **Conclusion**: 0.5 sayfa (yazılacak)

**Toplam**: 7-8 sayfa

## Timeline

- **Gün 1**: Abstract + Introduction
- **Gün 2**: Method + Experiments
- **Gün 3**: Discussion + Conclusion + Polish
- **Gün 4**: ArXiv Submit!

## ArXiv Submission

1. PDF'i compile et
2. https://arxiv.org → Login (uzunsukru hesabı)
3. "START NEW SUBMISSION"
4. Category: cs.LG (primary), cs.CL (secondary)
5. Upload: main.tex, references.bib, figures/
6. Submit!

**Hedef Tarih**: 3 Kasım 2025

