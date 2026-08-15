# Analysis: CGM and hQVM Genomics

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384. Repository: https://github.com/gyrogovernance/science

**Reproducibility:** `experiments/hqvm_cgm_genomics_1.py` through `_4.py`, `experiments/hqvm_cgm_genomics_common.py`, `experiments/hqvm_cgm_genomics_data_ingest.py`, `experiments/hqvm_cgm_genomics_run.py`. Combined output: `experiments/hqvm_cgm_genomics_results.txt`. External catalogs: `data/catalogs/genomics/` with provenance in `SOURCE.txt` and `MANIFEST.sha256`.

**Subject classes:** q-bio.GN; q-bio.QM; math-ph

**Keywords:** Common Governance Model, holonomic quantum virtual machine, genetic code, codon chart, reverse complement, fold map, Walsh-Hadamard, Square-Root Cluster Theorem, QuBEC, MaveDB, ClinVar

## Abstract

The Holonomic Quantum Virtual Machine (hQVM) is the executable form of the Common Governance Model (CGM) on a finite reversible transducer. The machine has 4096 reachable states, an 8-bit instruction alphabet, a 6-bit chirality register, and a Klein four-group family fiber. This analysis places nucleotide chemistry, codon translation, mutation transport, splice 4-mers, and codon-resolved mutational scans onto that machine. Objects that share a cardinality remain separate types.

A nucleotide is a 2-bit label. The 24 affine charts of `AGL(2,2)` split into three orbits of eight by which chemical involution has Hamming weight 2. Two named associations are retained on every later statistic: `orbit_elementary_axes`, where Watson-Crick and transition are independent weight-1 axes and transversion is the weight-2 composite, and `orbit_pair_inversion`, where Watson-Crick is the pair-diagonal complement `XOR 11`. Reverse-complement of a codon is an affine involution on all 24 charts. The palindromic fold `P` of the 8-bit intron is a different involution. The two maps commute on `orbit_pair_inversion` and fail to commute on `orbit_elementary_axes`.

The standard translation table has wreath automorphism order 2. Serine is a disconnected 6-codon fiber. Stop is a quotient-boundary class with leakage 0.85185. Ancestral and present codon states `(u, v)` on Omega satisfy `chirality = u XOR v` on all 4096 pairs. Transition-only generators reach 256 states against the Square-Root Cluster prediction 256 at rank 4. Synonymous, frame-preserving, and stop-preserving generators reach the full 4096-state bulk at rank 6. Sequence windows and MaveDB nested models are empirical layers on those frozen charts. Claim language distinguishes derived identities, mapping assumptions, and catalog tests, and labels each numerical claim as chart-invariant, orbit-dependent, or chart-dependent.

## 1. Scope and tiers of claim

### 1.1 Scope

CGM derives three spatial dimensions with six degrees of freedom. The chirality register of the hQVM carries one binary channel for each of those six degrees. The 6-bit codon payload and the 4096-state Omega carrier are algebraic objects inside that 3-space / 6-DoF framework. They are not extra spatial dimensions.

Four genomic objects share the machine and remain distinct.

A `CodonState` is a 6-bit payload obtained by concatenating three 2-bit nucleotide labels.

A `MutationTransport` is the XOR of two codon states.

A `ContextByte` is an 8-bit kernel byte. Two packings are kept: the bracket packing places two family bits around a 6-bit payload, and the linear 4-mer packing places four nucleotides so that the BU hinge bits 3 and 4 sit between bases 2 and 3.

A `GenomicCarrierPair` is an ancestral/present codon pair `(u, v)` read as a point of Omega.

Family phase in `K4` is unassigned until a lift map from 5-prime context, 3-prime context, strand, or 4-mer outer bits is chosen. Phenotype is excluded from that choice.

### 1.2 Tiers of claim

**Tier A (channel basis).** Identities that follow from kernel maps imported from `gyroscopic/hQVM` together with the affine nucleotide census. These include the 24-chart count, Watson-Crick / transition / transversion translations, reverse-complement affinity, `q_word6` packing consistency, fold `P` as the phase-pair reverse of `Analysis_hQVM_Wavefunction.md`, and `chirality(s(u,v)) = u XOR v`.

**Tier B (chart geometry of translation).** Fiber atlas, wreath automorphisms, Walsh-Hadamard degree energy, and N1-N5 nulls with `p_hat = (hits+1)/(N+1)`. Orbit summaries report min, mean, and max. The gate list omits minimum-tail chart selection across all 24 encodings.

**Tier C (sequence and effect).** Genome and splice-window occupation, nested OLS on MaveDB scores, chr22 phyloP after GC, CpG, and codon position, and a consequence-matched ClinVar panel. These tests use frozen charts from Tiers A and B.

### 1.3 External data

NCBI translation tables 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 21, 22, 23, 24, 25, 26, 29, 30, and 33 are frozen in `ncbi_genetic_codes.json`. Sequence windows use RefSeq CDS slices for *Escherichia coli* K-12 (GCF_000005845.2), *Saccharomyces cerevisiae* S288C (GCF_000146045.2), SARS-CoV-2 (GCF_009858895.2), GENCODE v47 annotation, and hg38 chr22. Effect models use MaveDB score tables and a GRCh38 ClinVar SNV panel for BRCA1, TP53, PTEN, MSH2, LDLR, CFTR, HBB, and PAH. SHA256 digests are recorded in `data/catalogs/genomics/MANIFEST.sha256`.

## 2. Nucleotide charts and metric orbits

The reference encoding is `A=00`, `G=01`, `T=10`, `C=11`. Watson-Crick is translation by `10` (weight 1). Transition is translation by `01` (weight 1). Transversion is translation by `11` (weight 2). The 24 affine encodings realize every assignment of the weight-2 involution to one of the three chemical involutions, eight encodings each. Chart-invariant counts are 24 distinct bijections, Watson-Crick translation on 24/24, transition on 24/24, transversion on 24/24, and `K4` span 2 on 24/24. The three chemical bipartitions R/Y, M/K, and S/W are linear bits on every chart.

`orbit_elementary_axes` is the eight encodings with Watson-Crick weight 1, transition weight 1, and transversion weight 2. `orbit_pair_inversion` is the eight encodings with Watson-Crick equal to `XOR 11`, which is the kernel pair-diagonal complement used by gate `F` and `epsilon`. The remaining eight encodings form `orbit_other`, in which transition carries weight 2. Both named orbits are reported on every later statistic.

Codon reverse-complement is an affine involution of rank 6 on all 24 charts. On the reference chart the translation part has weight 3. Fold `P` matches the four phase pairs `(0,7)`, `(1,6)`, `(2,5)`, `(3,4)` on 256/256 packed bytes. Fold disagreement over family times codon is the binomial histogram `(16, 64, 96, 64, 16)`. Payload rank of `fold+I` is 3. Payload rank of `RC+I` is 2. On the 8-bit maps both fold and reverse-complement are involutions. They commute on all 256 bytes of every `orbit_pair_inversion` chart and on 0/256 bytes of every `orbit_elementary_axes` chart. That commutator is orbit-dependent.

Gate `G8` certifies packing consistency: `q_word6` of `byte_from_family_micro(family, payload)` equals the predicted 6-bit transport, including the `L0` complement when family parity is odd, on 256/256 bytes and 24/24 charts. Mutation transport of a one-base neighbor pair equals the XOR of the two codon states on the reference chart.

## 3. Translation quotient and fiber atlas

The standard code occupies 64 codon labels with degeneracy `(4, 2, 2, 2, 2, 4, 2, 3, 2, 6, 1, 2, 4, 2, 6, 6, 4, 4, 1, 2, 3)` in amino-acid order `A` through `Y` then stop, summing to 64, with twenty amino-acid classes. The binomial `C(6,3)=20` is recorded as a multiplicity parallel with those twenty classes.

Serine is a 6-codon fiber with 2 connected components, transport rank 3, affine hull 8, density 0.75, cycle rank 3, 40 boundary edges, and leakage 0.74074. Stop is a 3-codon class with leakage 0.85185 and 23 boundary edges. Methionine and tryptophan are 1-codon classes with leakage 1. Chart-invariant geometry of the Hamming graph of synonymous one-base edges is independent of the nucleotide encoding. Transport rank and Krawtchouk moments of a fiber are chart-dependent through the bit packing and are reported on the reference chart.

Pointwise wreath automorphisms of the standard code number 2. One is the identity. The other is the third-position letter map that sends `A,C,G,T` to `A,T,G,C`, exchanging `C` and `T` at the wobble site. Edge-class automorphisms number 1. Stop-set automorphisms number 48. The wobble `K4` of last-base XOR translations preserves the standard code for 1 of 4 deltas on both named orbits. Mitochondrial tables 2, 3, 5, and 13 have pointwise Aut 4 and wobble occupancy 2. Bacterial/plastid table 11 matches the standard Aut lattice.

Walsh-Hadamard degree energy of the twenty-one class indicators has Plancherel sum 4096. On the standard code, cumulative weight through degree 2 is `S2=0.59961` on `orbit_elementary_axes`, `S2=0.63086` on `orbit_pair_inversion`, and `S2=0.54297` on `orbit_other`. N1 degeneracy-preserving shuffles give `p_hat=0.0050` on all three orbits with `N=200`. N2 first-two-base box shuffles give `p_hat=0.1045` on the elementary orbit, `0.0050` on pair inversion, and `0.9950` on `orbit_other`. N3 relabeling of equal fiber geometry leaves `S2` unchanged (`p_hat=1`). N4 stop-preserving shuffles give `p_hat=0.0123`. N5 transversion synonymous-edge count is 19 with `p_hat=0.0123`. These tails are orbit summaries. The gate list reports them without a minimum-p pass criterion.

## 4. Context lift, Omega pairing, and kernel theorems on genomic generators

Candidate family maps `eta` from a 2-bit context nucleotide through `GL(2,2)` and a 2-bit shift yield 48 nondegenerate reverse-complement covariant maps with occupancy `(1,1,1,1)` on each of the two named orbits. Dual charts are `B+ = (5-prime family, codon payload)` and `B- = (Watson-Crick of the 3-prime nucleotide, reverse-complement payload)`. Fold of `B+` equals `B-` on 0/1024 elementary duals and on 32/1024 pair-inversion duals. Mean Hamming distance is 4.0 on both orbits. Palindromic 5-prime / 3-prime pairs have family XOR 0.

Linear 4-mer packing unpacks on 256/256 tetramers. Equality with the bracket packing of `(first base as family, last three bases as codon)` holds on 4/256 tetramers. The two packings are distinct charts of the same 8-bit alphabet.

For ancestral state `u` and present state `v`, the Omega chirality of `(u, v)` equals `u XOR v` on 4096/4096 pairs, and `word6_to_pairdiag12` round-trips on 64/64 codon states. Stepping from `(u, u)` by the packed transport byte increments chirality by `q_word6` of that byte on 4096/4096 trials. Gate `F` sends `(u, u)` to `(u XOR 0x3F, u XOR 0x3F)` on 64/64 codon states. The `K4` orbit of a generic pair has size 4. The equality-horizon pair `(0, 0)` has orbit size strictly less than 4.

Lifted `WordSignature` composition with family frozen at 0 matches the two-byte word on 81/81 generator pairs, commutes on 9, and fails to commute on 72. The same ratios hold with family derived from the 5-prime nucleotide (1296/1296 homomorphism, 144 commute, 1152 noncommute). Additive `q_word6` holds on both family modes. Noncommutativity lives in the lifted signature, whose translation pair is swapped by odd parity.

Generator sets are packed one-base mutation bytes with all four family labels. Transition-only bytes number 12, have transport rank 4, predicted cluster size 256, and measured reach 256, with shell census `(16, 48, 48, 32, 48, 48, 16)`. Synonymous-only, frame-preserving, and stop-preserving sets have rank 6 and reach 4096, matching the full Omega census `(64, 384, 960, 1280, 960, 384, 64)`. Fiber-completeness fails on these restricted alphabets. One-base codon mutations occupy shells 1 and 2 only, with counts 384 and 192. Dicodon XOR occupation reproduces the Omega shell census.

QuBEC `Z1(lambda=1)` equals 4096. Shell populations sum to 4096.

Hydrogen-bond, stacking, and ATP hydrolysis energies are placed on the allometric chemical clock `E_a = kT / (2 Delta)` at `T=310.15 K`, giving `E_a = 0.645585 eV`. A 3 kcal/mol hydrogen bond is 0.2015 of `E_a`. A 2 kcal/mol stacking contact is 0.1343 of `E_a`. ATP at 7.3 kcal/mol is 0.4903 of `E_a`. These ratios are corpus associations on the aperture gap `Delta`.

## 5. Sequence windows

The 4-mer alphabet has fold-disagreement histogram `(16, 64, 96, 64, 16)` and mean 2.0 on both named orbits, matching the family-times-codon histogram. Dual-chart fold equality remains 0/1024 on the elementary orbit and 32/1024 on pair inversion.

CDS windows were scored on both orbits. *E. coli* K-12, 4000 records and 1,245,482 codons, has GC 0.519. Mean 4-mer fold disagreement is 1.990 (elementary) and 1.943 (pair inversion). Dual-chart Hamming means are 3.956 and 3.801. Mutation-to-consensus shells are 0.639 and 0.725. Codon-usage L1 distance to the GC multinomial is 0.528 with N1 `p_hat=0.012` on both orbits. Yeast S288C, 4000 records and 1,945,556 codons, has GC 0.397, L1 0.372, and N1 `p_hat=0.012`. SARS-CoV-2 CDS, 12 records and 14,161 codons, has GC 0.379, L1 0.448, and N1 `p_hat=0.012`. Human chr22 CDS from 300 GENCODE transcripts, 109,493 codons, has GC 0.585, L1 0.468, and N1 `p_hat=0.012`. Motif-preserving shuffles of 4-mer fold disagreement give `p_hat=1.000` on *E. coli* for both orbits, `0.704` and `1.000` on yeast, `0.012` and `1.000` on SARS-CoV-2, and `1.000` and `0.012` on chr22. Dicodon mean shells lie between 2.737 and 2.909, near the uniform dicodon mean 3.

GENCODE chr22 splice windows on the elementary orbit have donor mean fold disagreement 2.755 with GT/AG fraction 0.927 and shuffle `p_hat=0.012`, acceptor 1.781 with fraction 0.111 and `p_hat=1.000`, exon flank 1.977, and intron interior 1.957. Pair-inversion donor mean is 3.159 with the same GT/AG fraction and `p_hat=0.012`. Donor elevation relative to exon flank and intron interior is orbit-dependent in magnitude and chart-invariant in the GT/AG motif rate.

## 6. Path and effect

Nested ordinary least squares uses an intercept plus mutation-class and synonymous flags (M1), then amino-acid and codon-position and CpG terms (M2), then hQVM shell, parity, rank, and `WordSignature` translation weights (M3). Gene-style holdout is a random 20 percent of rows. Permutation shuffles the `WordSignature` weights inside the same design. A planted outcome linear in those weights on 270 one-base neighbors recovers `R2_M3=0.98648` against `R2_M1=0.00032`, holdout 0.98664, and permutation `p_hat=0.01235` (0/80).

Eight MaveDB score sets were parsed. Where nucleotide HGVS is absent, each protein substitution is mapped to a representative source codon and a minimum-Hamming destination codon in the destination amino-acid fiber. That map is an assumption. Under it, M3 adds little beyond M2. Representative values are `R2_M3=0.09255` on urn:mavedb:00000048-a-1 (`n=6995`) and `R2_M3=0.00837` on urn:mavedb:00000001-a-4 (`n=6699`). Permutation `p_hat=1` on all eight assays. The planted recovery and the assay increment are therefore different claims. The nested design recovers planted signature weights at `R2_M3=0.98648` with permutation `p_hat=0.01235`. The representative-codon embedding of these score tables yields permutation `p_hat=1`.

Chr22 phyloP at 4000 CDS codon starts, after GC, CpG, and genomic position modulo 3, has `R2_base=0.02377`. Adding codon-state shell, parity, and rank gives `R2=0.02519` on the elementary orbit and `0.02500` on pair inversion.

The ClinVar GRCh38 SNV panel reconstructs a codon pair from the HGVS nucleotide change on an `AAA` scaffold at the coding position modulo 3. Mean mutation-shell of pathogenic versus benign calls is 1.27864 versus 1.15571 for BRCA1 (`n=4090`), 1.28106 versus 1.16287 for TP53 (`n=951`), 1.31233 versus 1.11696 for PTEN (`n=536`), and similarly ordered on LDLR, CFTR, HBB, and PAH. MSH2 reverses the order (1.22815 pathogenic versus 1.15607 benign). This scaffold is a mapping assumption. Consequence matching is gene-restricted SNV, not a genome-wide search.

## 7. What would fail

The 24-chart census fails if some affine encoding does not make Watson-Crick a translation. Reverse-complement affinity fails if codon RC is nonlinear on any chart. Packing consistency fails if `q_word6` disagrees with the family-conditioned payload on any byte. Fold identity fails if `fold_map_d` disagrees with the phase-pair reverse. Omega pairing fails if chirality of `(u, v)` disagrees with `u XOR v`. Square-Root Cluster application to transition generators fails if reach disagrees with `(2^rank)^2`. WordSignature composition fails if the two-byte word disagrees with `compose_word_signatures`. Standard-code Aut order fails if a third wreath element preserves translation. Serine disconnectivity fails if the synonymous Hamming graph of serine is connected. Orbit retention fails if a later gate discards one named orbit by taking a minimum tail over all 24 charts.

Sequence and effect layers fail as empirical tests when GC-matched codon usage of a listed genome is typical of the N1 null, when donor 4-mer fold disagreement is typical of shuffled windows, or when a nucleotide-resolved MaveDB table with true codon context yields an M3 increment no larger than a permutation of hQVM labels. Those tests remain open wherever the catalog supplies only protein HGVS.

## 8. Scripts and gates

Script 1 reports the chart and translation quotient. Script 2 reports context lift, packings, Omega pairing, generator reach, lifted words, and the Delta-ruler association. Script 3 reports sequence and splice windows. Script 4 reports nested models, constraint, and the ClinVar panel. The combined run recorded 66/66 predeclared gates passing in `experiments/hqvm_cgm_genomics_results.txt`. Scripts print measurements and PASS/FAIL lines. Interpretive language is confined to this note.
