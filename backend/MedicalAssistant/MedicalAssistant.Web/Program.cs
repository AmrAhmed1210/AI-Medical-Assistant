using MedicalAssistant.Application.Services;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Persistance.Data.DbContexts;
using MedicalAssistant.Persistance.Data.SqlServer;
using MedicalAssistant.Persistance.Repositories;
using MedicalAssistant.Presentation.Hubs;
using MedicalAssistant.Services.MappingProfiles;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.Settings;
using MedicalAssistant.Web.Filters;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi.Models;
using System.Text;

namespace MedicalAssistant;

public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        builder.Logging.ClearProviders();
        builder.Logging.AddConsole();
        builder.Logging.AddDebug();

        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();

        builder.Services.AddSwaggerGen(c =>
        {
            c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
            {
                Name = "Authorization",
                Type = SecuritySchemeType.Http,
                Scheme = "Bearer",
                BearerFormat = "JWT",
                In = ParameterLocation.Header,
                Description = "Enter JWT token"
            });

            c.AddSecurityRequirement(new OpenApiSecurityRequirement
            {
                {
                    new OpenApiSecurityScheme
                    {
                        Reference = new OpenApiReference
                        {
                            Type = ReferenceType.SecurityScheme,
                            Id = "Bearer"
                        }
                    },
                    Array.Empty<string>()
                }
            });

            c.OperationFilter<SwaggerFileUploadFilter>();
        });

        var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
            ?? throw new InvalidOperationException("DefaultConnection is missing");

        builder.Services.AddDbContext<MedicalAssistantDbContext>(options =>
            options.UseSqlServer(connectionString, o => o.EnableRetryOnFailure()));

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
                    IssuerSigningKey = new SymmetricSecurityKey(
                        Encoding.UTF8.GetBytes(jwtKey))
                };

                options.Events = new JwtBearerEvents
                {
                    OnMessageReceived = context =>
                    {
                        var accessToken = context.Request.Query["access_token"];
                        var path = context.HttpContext.Request.Path;

                        if (!string.IsNullOrEmpty(accessToken) &&
                            path.StartsWithSegments("/hubs/notifications"))
                        {
                            context.Token = accessToken;
                        }

                        return Task.CompletedTask;
                    }
                };
            });

        builder.Services.AddAuthorization();

        builder.Services.AddSignalR();
        builder.Services.AddSingleton<IUserIdProvider, CustomUserIdProvider>();

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
        builder.Services.AddScoped<IPatientRecordService, PatientRecordService>();
        builder.Services.AddScoped<IPatientVisitService, PatientVisitService>();
        builder.Services.AddScoped<ISecretaryService, SecretaryService>();

        var pythonServiceUrl =
            builder.Configuration["AIService:Url"]
            ?? builder.Configuration["MEDICAL_AI_URL"]
            ?? "http://localhost:8000";

        builder.Services.AddHttpClient<IMedicalAiService, MedicalAiService>(client =>
        {
            var url = builder.Configuration["AIService:Url"]
                      ?? "http://localhost:8000";

            client.BaseAddress = new Uri(url);
            client.Timeout = TimeSpan.FromSeconds(60);
            client.DefaultRequestHeaders.Add("x-internal-token", "LuxuryMedicalAiSecretKey2026");
        });


        builder.Services.Configure<CloudinarySettings>(
            builder.Configuration.GetSection("CloudinarySettings"));

        builder.Services.AddAutoMapper(cfg => { }, typeof(DoctorProfile));

        var corsOrigins =
            builder.Configuration.GetSection("CorsOrigins").Get<string[]>()
            ?? new[]
            {
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:8081"
            };

        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowAll", policy =>
                policy.WithOrigins(corsOrigins)
                      .AllowAnyMethod()
                      .AllowAnyHeader()
                      .AllowCredentials());
        });

        var app = builder.Build();

        app.UseSwagger();
        app.UseSwaggerUI();
       app.UseHttpsRedirection();

        app.UseCors("AllowAll");

        app.UseAuthentication();
        app.UseAuthorization();

        app.MapControllers();
        app.MapHub<NotificationHub>("/hubs/notifications");

        using (var scope = app.Services.CreateScope())
        {
            var services = scope.ServiceProvider;
            var logger = services.GetRequiredService<ILogger<Program>>();

            try
            {
                logger.LogInformation("Applying SQL Server migrations...");

                var context = services.GetRequiredService<MedicalAssistantDbContext>();
                SqlServerMigrationBootstrap.Apply(context, logger);

                // Seed Admin User securely if not exists
                var adminUser = context.Set<MedicalAssistant.Domain.Entities.UserModule.User>()
                    .FirstOrDefault(u => u.Email == "admin@admin.com");

                if (adminUser == null)
                {
                    logger.LogInformation("Seeding Admin user...");
                    var hashedPassword = BCrypt.Net.BCrypt.HashPassword("12345678");

                    var newAdmin = new MedicalAssistant.Domain.Entities.UserModule.User
                    {
                        FullName = "Admin",
                        Email = "admin@admin.com",
                        PasswordHash = hashedPassword,
                        Role = "Admin",
                        IsActive = true,
                        IsDeleted = false,
                        CreatedAt = DateTime.UtcNow
                    };

                    context.Set<MedicalAssistant.Domain.Entities.UserModule.User>().Add(newAdmin);
                    context.SaveChanges();
                    logger.LogInformation("Admin user seeded successfully.");
                }
                else if (adminUser.FullName == "Hassan Mohamed")
                {
                    // Update existing seeded user to "Admin"
                    adminUser.FullName = "Admin";
                    context.SaveChanges();
                    logger.LogInformation("Admin user name updated from Hassan Mohamed to Admin.");
                }
            }
            catch (Exception ex)
            {
                logger.LogCritical(
                    ex,
                    "FATAL ERROR: Database migration failed!");
            }
        }

        app.Run();
    }
}
