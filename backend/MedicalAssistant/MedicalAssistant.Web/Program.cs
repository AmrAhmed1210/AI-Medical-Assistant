using MedicalAssistant.Application.Services;
// Force redeploy to apply database migrations for PostgreSQL
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Persistance.Data.DbContexts;
using MedicalAssistant.Persistance.Repositories;
using MedicalAssistant.Presentation.Hubs;
using MedicalAssistant.Services.MappingProfiles;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.Settings;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http.Features;
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

        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();

        // ── Allow large file uploads (up to 50 MB) ────────────────────────────
        builder.Services.Configure<FormOptions>(options =>
        {
            options.MultipartBodyLengthLimit = 52428800; // 50 MB
            options.ValueLengthLimit = int.MaxValue;
            options.MultipartHeadersLengthLimit = int.MaxValue;
        });
        builder.WebHost.ConfigureKestrel(options =>
        {
            options.Limits.MaxRequestBodySize = 52428800; // 50 MB
        });
        builder.Services.AddSwaggerGen(c =>
        {
            c.CustomSchemaIds(type => type.FullName);
            c.OperationFilter<MedicalAssistant.Web.Filters.SwaggerFileUploadFilter>();
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

        var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");

        builder.Services.AddDbContext<MedicalAssistantDbContext>(options =>
        {
            if (string.IsNullOrEmpty(connectionString))
            {
                throw new InvalidOperationException("Connection string 'DefaultConnection' not found.");
            }

            if (connectionString.Contains("postgresql") || connectionString.Contains("supabase") || connectionString.Contains("Host="))
            {
                options.UseNpgsql(connectionString, npgsqlOptions =>
                    npgsqlOptions.EnableRetryOnFailure(
                        maxRetryCount: 5,
                        maxRetryDelay: TimeSpan.FromSeconds(10),
                        errorCodesToAdd: null));
            }
            else
            {
                options.UseSqlServer(connectionString, sqlOptions =>
                    sqlOptions.EnableRetryOnFailure());
            }
        });

        var jwtKey = builder.Configuration["Jwt:Key"]
            ?? throw new InvalidOperationException("JWT Key is missing");

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

        // ── Cloudinary ────────────────────────────────────────────────────────
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
                policy.WithOrigins(
                          "http://localhost:3000",
                          "http://127.0.0.1:3000",
                          "http://localhost:5173",
                          "http://127.0.0.1:5173")
                      .AllowAnyMethod()
                      .AllowAnyHeader()
                      .AllowCredentials());
        });

        var app = builder.Build();

        // ── Phase 1: Run migrations (creates all tables in Supabase) ──────────────
        using (var scope = app.Services.CreateScope())
        {
            var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
            try
            {
                var context = scope.ServiceProvider.GetRequiredService<MedicalAssistantDbContext>();
                
                // --- STRATEGIC PATCH START: Run before everything else ---
                logger.LogInformation("🛠️ Starting Early Schema Patch (Pre-Migration)...");
                try 
                {
                    string[] patchSqls = {
                        "ALTER TABLE \"Patients\" ADD COLUMN IF NOT EXISTS \"UserId\" integer;",
                        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS \"UserId\" integer;",
                        "ALTER TABLE \"Users\" ADD COLUMN IF NOT EXISTS \"PhotoUrl\" text;",
                        "ALTER TABLE users ADD COLUMN IF NOT EXISTS \"PhotoUrl\" text;",
                        "ALTER TABLE \"Session\" ADD COLUMN IF NOT EXISTS \"Type\" text DEFAULT '';",
                        "ALTER TABLE \"Message\" ADD COLUMN IF NOT EXISTS \"AttachmentUrl\" text;",
                        "ALTER TABLE \"Message\" ADD COLUMN IF NOT EXISTS \"FileName\" text;",
                        "ALTER TABLE \"Message\" ADD COLUMN IF NOT EXISTS \"MessageType\" text DEFAULT 'text';",
                        "ALTER TABLE \"DoctorApplications\" ADD COLUMN IF NOT EXISTS \"PhotoUrl\" text;"
                    };

                    foreach (var sql in patchSqls)
                    {
                        try { await context.Database.ExecuteSqlRawAsync(sql); } catch { /* Silent fail */ }
                    }
                    logger.LogInformation("✅ Pre-migration patch complete.");
                }
                catch(Exception ex) { logger.LogWarning("⚠️ Early patch warning: {Msg}", ex.Message); }
                // --- STRATEGIC PATCH END ---

                logger.LogInformation("🔄 Applying database migrations...");
                await context.Database.MigrateAsync();
                logger.LogInformation("✅ Database migrations applied successfully.");
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "❌ An error occurred during DB initialization, but we will try to continue.");
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "❌ An error occurred while migrating the database.");
                // Note: On some local setups we might want to continue even if migration fails if we already have the DB
            }
        }

        // ── Phase 2: Seed default admin user ─────────────────────────────────────
        using (var scope = app.Services.CreateScope())
        {
            try
            {
                var context = scope.ServiceProvider.GetRequiredService<MedicalAssistantDbContext>();
                var adminEmail = "admin@medbook.com";
                
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
                    Console.WriteLine("Created Admin User: " + adminEmail);
                }
                else
                {
                    // Ensure existing admin is active and has correct role
                    admin.IsActive = true;
                    admin.IsDeleted = false;
                    admin.Role = "Admin";
                    // Reset password to default if needed (optional, but good for rescue)
                    admin.PasswordHash = BCrypt.Net.BCrypt.HashPassword("123456789");
                    context.Set<MedicalAssistant.Domain.Entities.UserModule.User>().Update(admin);
                    await context.SaveChangesAsync();
                    Console.WriteLine("Verified/Updated Admin User: " + adminEmail);
                }
                // ── Seeding Specialties ───────────────────────────────────────────
                var specialties = await context.Set<MedicalAssistant.Domain.Entities.DoctorsModule.Specialty>().ToListAsync();
                if (!specialties.Any())
                {
                    var initialSpecialties = new List<MedicalAssistant.Domain.Entities.DoctorsModule.Specialty>
                    {
                        new() { Name = "General Practice", NameAr = "طب عام" },
                        new() { Name = "Cardiology", NameAr = "أمراض القلب" },
                        new() { Name = "Dermatology", NameAr = "الأمراض الجلدية" },
                        new() { Name = "Neurology", NameAr = "أمراض المخ والأعصاب" },
                        new() { Name = "Orthopedics", NameAr = "جراحة العظام" },
                        new() { Name = "Pediatrics", NameAr = "طب الأطفال" },
                        new() { Name = "Psychiatry", NameAr = "الطب النفسي" },
                        new() { Name = "Surgery", NameAr = "الجراحة العامة" }
                    };
                    await context.Set<MedicalAssistant.Domain.Entities.DoctorsModule.Specialty>().AddRangeAsync(initialSpecialties);
                    await context.SaveChangesAsync();
                    Console.WriteLine("Seeded initial specialties.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("⚠️ Seeding failed: " + ex.Message);
            }
        }

        app.UseSwagger();
        app.UseSwaggerUI(c =>
        {
            c.SwaggerEndpoint("/swagger/v1/swagger.json", "Medical Assistant API V1");
            c.RoutePrefix = "swagger"; 
        });

        // NOTE: HttpsRedirection is disabled because the server runs on plain HTTP (http://0.0.0.0:5194)
        // app.UseHttpsRedirection();
        
        app.UseCors("AllowAll");
        app.UseAuthentication();
        app.UseAuthorization();
        app.MapControllers();
        app.MapHub<NotificationHub>("/hubs/notifications");

        app.Run();
    }
}