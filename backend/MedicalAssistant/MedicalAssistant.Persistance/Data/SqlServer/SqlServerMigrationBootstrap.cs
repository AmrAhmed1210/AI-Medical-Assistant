using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace MedicalAssistant.Persistance.Data.SqlServer;

public static class SqlServerMigrationBootstrap
{
    public static void Apply(MedicalAssistantDbContext context, ILogger logger)
    {
        var pending = context.Database.GetPendingMigrations().ToList();
        if (pending.Count == 0)
        {
            logger.LogInformation("SQL Server schema is up to date.");
            return;
        }

        logger.LogInformation(
            "Applying {Count} SQL Server migration(s): {Migrations}",
            pending.Count,
            string.Join(", ", pending));

        context.Database.Migrate();

        logger.LogInformation("SQL Server migrations applied successfully.");
    }
}
