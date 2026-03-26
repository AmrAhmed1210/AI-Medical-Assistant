export const Skeleton = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`animate-pulse rounded-xl bg-gray-200/80 ${className}`} {...props} />
)