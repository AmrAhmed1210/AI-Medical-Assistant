export const Table = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableElement>) => (
  <div className="overflow-x-auto"><table className={`w-full ${className}`} {...props} /></div>
)

export const TableHeader = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableSectionElement>) => (
  <thead className={`bg-gray-50/80 ${className}`} {...props} />
)

export const TableRow = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableRowElement>) => (
  <tr className={`border-b border-gray-100 hover:bg-gray-50/50 transition-colors ${className}`} {...props} />
)

export const TableHead = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableCellElement>) => (
  <th className={`px-4 py-4 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider ${className}`} {...props} />
)

export const TableCell = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableCellElement>) => (
  <td className={`px-4 py-4 text-sm text-gray-700 ${className}`} {...props} />
)