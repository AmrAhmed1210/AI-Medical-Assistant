using MedicalAssistant.Application.Services;
using MedicalAssistant.Persistance.Data.DbContexts;
using MedicalAssistant.Services.MappingProfiles;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Web.Extensions;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant;

public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // --- 1. Add services to the container ---
        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();
        
        // --- 2. Add DbContext ---
        builder.Services.AddDbContext<MedicalAssistantDbContext>(options =>
        {
            options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection"));
        });

        // --- 3. Register Modules ---
        builder.Services.AddPatientModule();


        builder.Services.AddScoped<IAuthService, AuthService>();

        // --- 4. AutoMapper Configuration ---
        builder.Services.AddAutoMapper(cfg =>
        {
            cfg.AddProfile<DoctorProfile>();
            // إذا قمت بإنشاء AdminProfile لاحقاً، أضفه هنا:
            // cfg.AddProfile<AdminProfile>(); 
        }, typeof(DoctorProfile).Assembly);

        // --- 5. Dependency Injection (Service Registration) ---
        // خدمات الأطباء والمراجعات
        builder.Services.AddScoped<IDoctorService, DoctorService>();
        builder.Services.AddScoped<IReviewService, ReviewService>();

        // تسجيل خدمة الأدمن الجديدة (السطر المطلوب)
        builder.Services.AddScoped<IAdminService, AdminService>();

        // --- 6. CORS Policy ---
        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowAll", policy =>
            {
                policy.AllowAnyOrigin()
                      .AllowAnyMethod()
                      .AllowAnyHeader();
            });
        });

        var app = builder.Build();

        // --- 7. Configure the HTTP request pipeline ---
        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        app.UseHttpsRedirection();

        app.UseCors("AllowAll"); // Enable CORS for Frontend

        app.UseAuthorization();

        // ربط الـ Controllers بالـ Routes
        app.MapControllers();

        app.Run();
    }
}