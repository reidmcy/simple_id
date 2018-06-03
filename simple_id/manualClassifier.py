import getkey
import os.path
import pandas

target = 'data/some.tsv'

output = 'data/classified.csv'

def writeLine(wosID, val):
    with open(output, 'a') as f:
        f.write('{},{}\n'.format(wosID, val))

def main():
    if not os.path.isfile(output):
        with open(output, 'w') as f:
            f.write('wos_id,isSoftware\n')

    dfOut = pandas.read_csv(output)
    dfIn = pandas.read_csv(target, sep = '\t')

    processedkeys = set(dfOut['wos_id'])

    newData = {
        'wos_id' : [],
        'isSoftware' : [],
    }

    count = 0
    total = len([w for w in dfIn['wos_id'] if w not in processedkeys])
    for i, row in dfIn.iterrows():
        if row['wos_id'] in processedkeys:
            continue
        count += 1

        print("\n" * 80)
        print("Paper number {} of {}, from {}".format(count, total, row['source']))
        print()
        print(row['title'])
        print()
        print(row['abstract'])

        key = getkey.getkey()

        while key not in ['y', 'n', 'e']:
            print("Invalid key Pressed")
            key = getkey.getkey()

        if key == 'y':
            val = 1
        elif key == 'n':
            val = 0
        elif key == 'e':
            print("Exiting")
            break
        else:
            print("Somthing bad happend")
            import pdb
            pdb.set_trace()
        writeLine(row['wos_id'], val)
    print("Done")
if __name__ == '__main__':
    main()
