from graphviz import Digraph

dot = Digraph()
# imaging nodes
dot.node('A', 'CHUV DICOMs')
dot.node('B', 'BIDSCoiner')
dot.node('C', 'NonCHUVCoiner')
dot.node('F', 'RTS2BIDS')
dot.node('D', 'Convert to Nifti')
dot.node('E', 'Relevant Images')
dot.node('G', 'Segmentations & Dose')
dot.node('H', 'Filter/Register')
dot.node('I', 'nnUNet Resegmentation')
dot.node('J', 'Preprocessed Dataset')

dot.edges(['CD', 'BD', 'DE', 'AF', 'FG', 'HI', 'IJ'])
dot.edge('A', 'C', label='Manual Selection')
dot.edge('A', 'B', label='Automatic Selection')
dot.edge('E', 'H', label='Criterion Selection')
dot.edge('G', 'H', label='Criterion Selection')

dot.render('flowchart', format='png')