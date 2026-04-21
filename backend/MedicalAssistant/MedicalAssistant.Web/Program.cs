using MedicalAssistant.Application.Services;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Persistance.Data.DbContexts;
using MedicalAssistant.Persistance.Repositories;
using MedicalAssistant.Presentation.Hubs;
using MedicalAssistant.Services.MappingProfiles;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
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

        // ── Database ─────────────────────────────────────────────────────────
        builder.Services.AddDbContext<MedicalAssistantDbContext>(options =>
            options.UseSqlServer(
                builder.Configuration.GetConnectionString("DefaultConnection"),
                sqlOptions => sqlOptions.EnableRetryOnFailure()));

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
                    IssuerSigningKey = new SymmetricSecurityKey(
                        Encoding.UTF8.GetBytes(jwtKey))
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
                    },
                    OnTokenValidated = async context =>
                    {
                        var email = context.Principal?.FindFirst(ClaimTypes.Email)?.Value;
                        var role = context.Principal?.FindFirst(ClaimTypes.Role)?.Value;

                        if (string.IsNullOrWhiteSpace(email) || string.IsNullOrWhiteSpace(role))
                        {
                            context.Fail("Invalid account.");
                            return;
                        }

                        var dbContext = context.HttpContext.RequestServices.GetRequiredService<MedicalAssistantDbContext>();

                        if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
                        {
                            var patient = await dbContext.Patients.FirstOrDefaultAsync(p => p.Email.ToLower() == email.ToLower());
                            if (patient == null || !patient.IsActive)
                            {
                                context.Fail("Your account is inactive.");
                            }

                            return;
                        }

                        var user = await dbContext.Users.FirstOrDefaultAsync(u => u.Email.ToLower() == email.ToLower());
                        if (user == null || !user.IsActive || user.IsDeleted)
                        {
                            context.Fail("Your account is inactive.");
                        }
                    }
                };
            });

        builder.Services.AddAuthorization();
        builder.Services.AddSignalR();
        builder.Services.AddSingleton<Microsoft.AspNetCore.SignalR.IUserIdProvider, CustomUserIdProvider>();

        // ── Repositories & Unit of Work ───────────────────────────────────────
        builder.Services.AddScoped<IUnitOfWork, UnitOfWork>();
        builder.Services.AddScoped<IAppointmentRepository, AppointmentRepository>();
        builder.Services.AddScoped<IDoctorRepository, DoctorRepository>();
        builder.Services.AddScoped<IPatientRepository, PatientRepository>();
        builder.Services.AddScoped<IReviewRepository, ReviewRepository>();

        // ── Services ──────────────────────────────────────────────────────────
        builder.Services.AddScoped<IPatientService, PatientService>();
        builder.Services.AddScoped<IAppointmentService, AppointmentService>();
        builder.Services.AddScoped<IDoctorService, DoctorService>();
        builder.Services.AddScoped<IReviewService, ReviewService>();
        builder.Services.AddScoped<IAuthService, AuthService>();
        builder.Services.AddScoped<IAdminService, AdminService>();
        builder.Services.AddScoped<INotificationService, NotificationService>();

        // ── AutoMapper ────────────────────────────────────────────────────────
        builder.Services.AddAutoMapper(cfg =>
        {
            cfg.AddProfile<DoctorProfile>();
            cfg.AddProfile<ReviewMappingProfile>();
            cfg.AddProfile<AdminProfile>();
        });

        // ── CORS ──────────────────────────────────────────────────────────────
        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowAll", policy =>
                policy.AllowAnyOrigin()
                      .AllowAnyMethod()
                      .AllowAnyHeader());
        });

        var app = builder.Build();
        
        // ── Seeding Admin User ────────────────────────────────────────────────
        try 
        {
            using (var scope = app.Services.CreateScope())
            {
                var context = scope.ServiceProvider.GetRequiredService<MedicalAssistantDbContext>();
                var adminEmail = "hassanmohamed5065@gmail.com";
                
                var adminExists = await context.Set<MedicalAssistant.Domain.Entities.UserModule.User>()
                    .AnyAsync(u => u.Email == adminEmail);

                if (!adminExists)
                {
                    var admin = new MedicalAssistant.Domain.Entities.UserModule.User
                    {
                        FullName = "Hassan Mohamed",
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
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("⚠️ Seeding failed: " + ex.Message);
        }

        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        app.UseHttpsRedirection();
        app.UseCors("AllowAll");
        app.UseAuthentication();
        app.UseAuthorization();
        app.MapControllers();
        app.MapHub<NotificationHub>("/hubs/notifications");
        app.Urls.Add("http://0.0.0.0:5194");
        app.Run();
    }
}
