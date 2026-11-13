import * as React from "react";

import { cn } from "./utils";

function Card({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div data-slot="card" className={cn("bg-card text-card-foreground flex flex-col gap-6 rounded-xl border", className)} {...props} />
  );
}

function CardContent({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="card-content" className={cn("px-6 [&:last-child]:pb-6", className)} {...props} />;
}

export { Card, CardContent };
