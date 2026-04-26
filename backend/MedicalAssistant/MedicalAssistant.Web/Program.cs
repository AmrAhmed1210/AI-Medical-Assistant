using MedicalAssistant.Application.Services;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Persistance.Data.DbContexts;
using MedicalAssistant.Persistance.Repositories;
using MedicalAssistant.Presentation.Hubs;
using MedicalAssistant.Services.MappingProfiles;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.Settings;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using System.Security.Claims;
using System.Text;

namespace MedicalAssistant;

public class Program
{
    public static async Task Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // ── Controllers & Swagger ────────────────────────────────────────────
        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen(c =>
        {
            c.AddSecurityDefinition("Bearer", new Microsoft.OpenApi.Models.OpenApiSecurityScheme
            {
                Name = "Authorization",
                Type = Microsoft.OpenApi.Models.SecuritySchemeType.Http,
                Scheme = "Bearer",
                BearerFormat = "JWT",
                In = Microsoft.OpenApi.Models.ParameterLocation.Header,
                Description = "Enter your JWT token"
            });
            c.AddSecurityRequirement(new Microsoft.OpenApi.Models.OpenApiSecurityRequirement
            {
                {
                    new Microsoft.OpenApi.Models.OpenApiSecurityScheme
                    {
                        Reference = new Microsoft.OpenApi.Models.OpenApiReference
                        {
                            Type = Microsoft.OpenApi.Models.ReferenceType.SecurityScheme,
                            Id = "Bearer"
                        }
                    },
                    Array.Empty<string>()
                }
            });
        });

        // ── Database (PostgreSQL Detection) ──────────────────────────────────
        var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");
        bool isPostgres = connectionString?.Contains("Host=") == true || connectionString?.Contains("User Id=") == true;

        builder.Services.AddDbContext<MedicalAssistantDbContext>(options =>
        {
            if (isPostgres)
            {
                options.UseNpgsql(connectionString, o => o.EnableRetryOnFailure());
            }
            else
            {
                options.UseSqlServer(connectionString, o => o.EnableRetryOnFailure());
            }
        });

        // ── JWT Authentication ────────────────────────────────────────────────
        var jwtKey = builder.Configuration["Jwt:Key"]
            ?? throw new InvalidOperationException("JWT Key is missing in appsettings.json");

        builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
            .AddJwtBearer(options =>
            {
                options.TokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuer = true,
                    ValidateAudience = true,
                    ValidateLifetime = true,
                    ValidateIssuerSigningKey = true,
                    ValidIssuer = builder.Configuration["Jwt:Issuer"],
                    ValidAudience = builder.Configuration["Jwt:Audience"],
                    IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey))
                };
                options.Events = new JwtBearerEvents
                {
                    OnMessageReceived = context =>
                    {
                        var accessToken = context.Request.Query["access_token"];
                        var path = context.HttpContext.Request.Path;
                        if (!string.IsNullOrEmpty(accessToken) && path.StartsWithSegments("/hubs/notifications"))
                        {
                            context.Token = accessToken;
                        }
                        return Task.CompletedTask;
                    }
                };
            });

        builder.Services.AddAuthorization();
        builder.Services.AddSignalR();
        builder.Services.AddSingleton<Microsoft.AspNetCore.SignalR.IUserIdProvider, CustomUserIdProvider>();

        // ── Repositories & Services ───────────────────────────────────────────
        builder.Services.AddScoped<IUnitOfWork, UnitOfWork>();
        builder.Services.AddScoped<IAppointmentRepository, AppointmentRepository>();
        builder.Services.AddScoped<IDoctorRepository, DoctorRepository>();
        builder.Services.AddScoped<IPatientRepository, PatientRepository>();
        builder.Services.AddScoped<IReviewRepository, ReviewRepository>();
        builder.Services.AddScoped<IPatientService, PatientService>();
        builder.Services.AddScoped<IAppointmentService, AppointmentService>();
        builder.Services.AddScoped<IDoctorService, DoctorService>();
        builder.Services.AddScoped<IReviewService, ReviewService>();
        builder.Services.AddScoped<ISessionService, SessionService>();
        builder.Services.AddScoped<IMessageService, MessageService>();
        builder.Services.AddScoped<IAuthService, AuthService>();
        builder.Services.AddScoped<IAdminService, AdminService>();
        builder.Services.AddScoped<INotificationService, NotificationService>();
        builder.Services.AddScoped<IPhotoService, PhotoService>();

        builder.Services.Configure<CloudinarySettings>(builder.Configuration.GetSection("CloudinarySettings"));

        builder.Services.AddAutoMapper(cfg =>
        {
            cfg.AddProfile<DoctorProfile>();
            cfg.AddProfile<ReviewMappingProfile>();
            cfg.AddProfile<AdminProfile>();
        });

        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowAll", policy =>
                policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
        });

        var app = builder.Build();

        // ── Database Initialization (The Heart of the Fix) ───────────────────
        using (var scope = app.Services.CreateScope())
        {
            var services = scope.ServiceProvider;
            var logger = services.GetRequiredService<ILogger<Program>>();
            var context = services.GetRequiredService<MedicalAssistantDbContext>();

            // 1. Aggressive PostgreSQL Schema Patching
            if (!context.Database.IsSqlServer())
            {
                Console.WriteLine("🚀 Starting Fortified PostgreSQL Schema Patching...");
                try
                {
                    var patchSql = @"
                        DO $$ 
                        BEGIN 
                            -- Ensure columns exist in Messages/Message table
                            FOR tab IN SELECT tablename FROM pg_tables WHERE tablename IN ('Messages', 'Message', 'messages', 'message')
                            LOOP
                                EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS ""SenderName"" text', tab);
                                EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS ""SenderPhotoUrl"" text', tab);
                                EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS ""Type"" integer DEFAULT 0', tab);
                                EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS ""IsSystem"" boolean DEFAULT false', tab);
                            END LOOP;

                            -- Ensure columns exist in Sessions/Session table
                            FOR tab IN SELECT tablename FROM pg_tables WHERE tablename IN ('Sessions', 'Session', 'sessions', 'session')
                            LOOP
                                EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS ""Type"" text DEFAULT ''''', tab);
                            END LOOP;

                            -- Ensure columns exist in Users/users table
                            FOR tab IN SELECT tablename FROM pg_tables WHERE tablename IN ('Users', 'users')
                            LOOP
                                EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS ""PhotoUrl"" text', tab);
                                EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS ""FullName"" text', tab);
                            END LOOP;
                        END $$;";

                    await context.Database.ExecuteSqlRawAsync(patchSql);
                    Console.WriteLine("✅ PostgreSQL Schema Patching Applied Successfully");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ Schema Patch Warning: {ex.Message}");
                }
            }

            // 2. Try Migrations (Independent)
            try
            {
                Console.WriteLine("🔄 Running Migrations...");
                await context.Database.MigrateAsync();
                Console.WriteLine("✅ Database Migrations completed.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Migration failed/skipped (Non-critical due to patch): {ex.Message}");
            }

            // 3. Seeding Admin (Independent)
            try
            {
                var adminEmail = "hassanmohamed5065@gmail.com";
                var admin = await context.Set<MedicalAssistant.Domain.Entities.UserModule.User>()
                    .FirstOrDefaultAsync(u => u.Email == adminEmail);

                if (admin == null)
                {
                    admin = new MedicalAssistant.Domain.Entities.UserModule.User
                    {
                        FullName = "Admin",
                        Email = adminEmail,
                        PasswordHash = BCrypt.Net.BCrypt.HashPassword("123456789"),
                        Role = "Admin",
                        IsActive = true,
                        CreatedAt = DateTime.UtcNow
                    };
                    await context.Set<MedicalAssistant.Domain.Entities.UserModule.User>().AddAsync(admin);
                    await context.SaveChangesAsync();
                    Console.WriteLine("✅ Admin seeded successfully.");
                }
            }
            catch (Exception ex) 
            { 
                Console.WriteLine($"⚠️ Seeding failed: {ex.Message}"); 
            }
        }

        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        app.UseCors("AllowAll");
        app.UseAuthentication();
        app.UseAuthorization();
        app.MapControllers();
        app.MapHub<NotificationHub>("/hubs/notifications");

        // Railway Port Fix
        var port = Environment.GetEnvironmentVariable("PORT") ?? "8080";
        app.Run($"http://0.0.0.0:{port}");
    }
}
